"""Canonical train, validate, and predict workflow with fast threshold search and streaming inference."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .blocking import BlockingStore
from .features import FEATURE_NAMES, feature_batch
from .model import PairModel
from .output import write_submission_from_store_streaming, write_submission_stream

LOGGER = logging.getLogger(__name__)


# Keep the production candidate cap aligned with BlockingStore.TOP_K.
# 96 is the measured-recall optimization point to validate on the real block test.
DEFAULT_TOP_K = 96
CACHE_SCHEMA_VERSION = "2026-09-26-entity-resolution-v5-recall"


def _module_signature(*names: str) -> dict[str, str]:
    """Hash the code modules that affect candidate generation/features.

    File size/mtime alone is insufficient for SQLite cache validity: a code
    edit can leave both values unchanged in some sync/build environments.
    """
    result: dict[str, str] = {}
    base = Path(__file__).resolve().parent
    for name in names:
        path = base / name
        if path.exists():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:20]
    return result


@dataclass(frozen=True)
class MatrixFiles:
    x_path: Path
    y_path: Path | None
    pairs_path: Path
    rows: int

    def x(self, mode: str = "r") -> np.ndarray | np.memmap:
        if self.rows == 0:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        return np.memmap(self.x_path, dtype=np.float32, mode=mode, shape=(self.rows, len(FEATURE_NAMES)))

    def y(self) -> np.ndarray | np.memmap | None:
        if self.y_path is None:
            return None
        if self.rows == 0:
            return np.empty((0,), dtype=np.int8)
        return np.memmap(self.y_path, dtype=np.int8, mode="r", shape=(self.rows,))


def _assert_materialized(paths: Sequence[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required dataset file not found: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            first = f.readline().strip()
        if first == "version https://git-lfs.github.com/spec/v1":
            raise RuntimeError(
                f"Dataset file {path} is a Git-LFS pointer, not the materialized challenge data. "
                "Fetch/materialize the official challenge dataset before running the pipeline."
            )


def _dataset_paths(directory: str | Path, split: str) -> tuple[Path, Path, Path]:
    root = Path(directory)
    return tuple(root / f"{split}_source{source}.tsv" for source in (1, 2, 3))  # type: ignore[return-value]


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _store_signature(data_dir: str | Path, split: str, top_k: int, with_truth: bool) -> dict:
    root = Path(data_dir)
    source_paths = [root / f"{split}_source{i}.tsv" for i in (1, 2, 3)]
    sig = {
        "split": split,
        "top_k": int(top_k),
        "with_truth": bool(with_truth),
        "source_files": {p.name: _file_signature(p) for p in source_paths},
        "truth_file": _file_signature(root / "train_ground_truth.tsv") if with_truth and (root / "train_ground_truth.tsv").exists() else None,
        "blocking_config": {
            "minhash_permutations": BlockingStore.MINHASH_PERMUTATIONS,
            "minhash_bands": BlockingStore.MINHASH_BANDS,
            "token_df_limit": BlockingStore.TOKEN_DF_LIMIT,
            "key_freq_max_source": BlockingStore.KEY_FREQ_MAX_SOURCE,
            "key_freq_max_target": BlockingStore.KEY_FREQ_MAX_TARGET,
            "high_freq_rescue_top": BlockingStore.HIGH_FREQ_RESCUE_TOP,
            "high_freq_rescue_max_block": BlockingStore.HIGH_FREQ_RESCUE_MAX_BLOCK,
        },
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "code_signatures": _module_signature(
            "blocking.py",
            "preprocessing.py",
            "features.py",
        ),
    }
    return sig


def _build_store(data_dir: str | Path, split: str, database: Path, top_k: int, with_truth: bool) -> BlockingStore:
    t_start = time.perf_counter()
    source1, source2, source3 = _dataset_paths(data_dir, split)
    expected = _store_signature(data_dir, split, top_k, with_truth)
    _assert_materialized([source1, source2, source3] + ([Path(data_dir) / "train_ground_truth.tsv"] if with_truth else []))
    store = BlockingStore(database, top_k=top_k)

    # Reuse ONLY when the database was built with the exact same inputs/configuration.
    # This prevents silent stale-candidate reuse when TOP_K or data/config changes.
    try:
        row = store.connection.execute("SELECT value FROM pipeline_meta WHERE key='signature'").fetchone()
        existing = json.loads(row[0]) if row else None
        has_final = store.connection.execute("SELECT COUNT(*) FROM final_candidates").fetchone()[0] > 0
    except sqlite3.OperationalError:
        existing, has_final = None, False

    if has_final and existing == expected:
        store.diagnostics = {"final": {"recall": store.recall_ceiling()[0]}} if with_truth else {}
        LOGGER.info("Reusing validated blocking store at %s", database)
        return store

    if has_final:
        LOGGER.info("Invalid/stale blocking cache at %s; rebuilding", database)

    store.reset()
    source_count = store.build_source_index(source1)
    if with_truth:
        store.add_truth(Path(data_dir) / "train_ground_truth.tsv")
    store.add_targets_and_retrieve((source2, source3))
    store.finalize_candidates()

    store.connection.execute("CREATE TABLE IF NOT EXISTS pipeline_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    store.connection.execute("INSERT OR REPLACE INTO pipeline_meta(key, value) VALUES ('signature', ?)", (json.dumps(expected, sort_keys=True),))
    store.connection.commit()

    candidates, average = store.candidate_summary()
    diag = store.diagnostics
    LOGGER.info("Blocking complete: %d Source-1 records, %d pairs (%.3f/entity) in %.1fs",
                source_count, candidates, average, time.perf_counter() - t_start)
    if "raw_blocking" in diag:
        raw = diag["raw_blocking"]
        LOGGER.info("  Raw blocking recall: %.4f%% (%d/%d), CMS: %.4f%%",
                     100 * raw["recall"], raw["retrieved"], raw["total"],
                     100 * raw.get("complete_match_set_recall", 0))
    if "final" in diag:
        fin = diag["final"]
        LOGGER.info("  Final recall: %.4f%% (%d/%d), CMS: %.4f%%, zero-true S1s: %.4f%%",
                     100 * fin["recall"], fin["retrieved"], fin["total"],
                     100 * fin.get("complete_match_set_recall", 0),
                     100 * fin.get("s1_with_zero_true_candidates", 0))
    if "final_stats" in diag:
        fs = diag["final_stats"]
        LOGGER.info("  Candidate stats: avg=%.1f, p50=%d, p95=%d, p99=%d, max=%d",
                     fs["avg_candidates"], fs.get("p50", 0), fs.get("p95", 0),
                     fs.get("p99", 0), fs["max_candidates"])
    return store


def _materialize(store: BlockingStore, scratch: Path, name: str, where: str = "", parameters: Sequence[object] = (), labels: bool = True, subsample: bool = False) -> MatrixFiles:
    if labels and subsample:
        # Keep all positives, plus hard/medium/easy negatives via smart sampling.
        # All negatives with similarity > 0.55 (hard), top-4 ranked (medium), plus 10% random (easy).
        cond = "(truth.target_id IS NOT NULL OR c.similarity > 0.55 OR c.rank <= 4 OR (length(s.name) + length(t.name) + c.rank) % 10 = 0)"
        where = f"({where}) AND {cond}" if where else cond
    rows = store.feature_count(where, parameters)
    x_path, pairs_path = scratch / f"{name}.features.f32", scratch / f"{name}.pairs.tsv"
    y_path = scratch / f"{name}.labels.i8" if labels else None
    if rows:
        x = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(rows, len(FEATURE_NAMES)))
        y = None if y_path is None else np.memmap(y_path, dtype=np.int8, mode="w+", shape=(rows,))
    else:
        x_path.write_bytes(b"")
        if y_path is not None:
            y_path.write_bytes(b"")
        x = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        y = None if y_path is None else np.empty((0,), dtype=np.int8)
    written = 0
    cursor = store.feature_rows(where, parameters)
    with pairs_path.open("w", encoding="utf-8", newline="") as pairs_file:
        while True:
            batch = []
            try:
                for _ in range(25_000):
                    batch.append(next(cursor))
            except StopIteration:
                pass
            if not batch:
                break
            x[written:written + len(batch)] = feature_batch([row[:-1] for row in batch])
            if y is not None:
                y[written:written + len(batch)] = [row[-1] for row in batch]
            pairs_file.writelines(f"{row[0]}\t{row[1]}\n" for row in batch)
            written += len(batch)
    x.flush()
    if y is not None:
        y.flush()
    LOGGER.info("Materialized %s: %d rows (%d features)", name, rows, len(FEATURE_NAMES))
    return MatrixFiles(x_path, y_path, pairs_path, rows)


def _batched_predict_proba(model: PairModel, matrix: MatrixFiles, batch_size: int = 100_000) -> np.ndarray:
    """Predict in chunks to avoid blowing up memory with massive prediction arrays."""
    x = matrix.x()
    rows = matrix.rows
    probs = np.zeros(rows, dtype=np.float32)
    offset = 0
    while offset < rows:
        end = min(rows, offset + batch_size)
        probs[offset:end] = model.predict_proba(x[offset:end])
        offset = end
    return probs


def _countries(store: BlockingStore) -> tuple[str, str]:
    values = list(store.connection.execute("SELECT country, COUNT(*) FROM source1 WHERE country<>'' GROUP BY country ORDER BY COUNT(*) DESC, country"))
    if len(values) < 2:
        raise ValueError("Country holdout requires at least two countries")
    return values[0][0], values[1][0]


def _fast_tune_threshold(
    store: BlockingStore,
    matrix: MatrixFiles,
    probabilities: np.ndarray,
    where: str,
    parameters: Sequence[object],
) -> tuple[float, float]:
    """Exact threshold optimization while avoiding giant Python string lists.

    The pair stream and Source-1 stream are both ordered by entity ID, so the
    per-pair Source-1 index is written directly to a compact int32 memmap.
    """
    entity_count = store.connection.execute(
        f"SELECT COUNT(*) FROM source1 s WHERE {where}", parameters
    ).fetchone()[0]
    if entity_count == 0:
        return 0.0, 1.0

    tc_arr = np.zeros(entity_count, dtype=np.int32)
    truth_cursor = store.connection.execute(
        f"SELECT t.source1_id, COUNT(*) FROM truth t JOIN source1 s ON s.entity_id=t.source1_id WHERE {where} GROUP BY t.source1_id ORDER BY t.source1_id",
        parameters
    )
    entity_cursor = iter(store.connection.execute(
        f"SELECT entity_id FROM source1 s WHERE {where} ORDER BY entity_id", parameters
    ))
    current = next(entity_cursor, None)
    current_sid = current[0] if current else None
    idx = 0
    for sid, count in truth_cursor:
        while current_sid is not None and current_sid < sid:
            idx += 1
            nxt = next(entity_cursor, None)
            current_sid = nxt[0] if nxt else None
        if current_sid == sid:
            tc_arr[idx] = int(count)

    idx_path = matrix.pairs_path.with_suffix(".s1idx.i4")
    s1_indices = np.memmap(idx_path, dtype=np.int32, mode="w+", shape=(matrix.rows,)) if matrix.rows else np.empty((0,), dtype=np.int32)
    entity_cursor = iter(store.connection.execute(
        f"SELECT entity_id FROM source1 s WHERE {where} ORDER BY entity_id", parameters
    ))
    current = next(entity_cursor, None)
    current_sid = current[0] if current else None
    entity_idx = 0
    for row_idx, line in enumerate(matrix.pairs_path.open("r", encoding="utf-8")):
        sid = line.split("\t", 1)[0]
        while current_sid is not None and current_sid < sid:
            entity_idx += 1
            current = next(entity_cursor, None)
            current_sid = current[0] if current else None
        if current_sid != sid:
            raise AssertionError(f"Pair stream S1 ID {sid!r} is not aligned with validation Source-1 ordering")
        s1_indices[row_idx] = entity_idx
    if matrix.rows:
        s1_indices.flush()

    labels = matrix.y()
    is_true_arr = np.asarray(labels, dtype=bool) if labels is not None else np.zeros(matrix.rows, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if len(probabilities) != matrix.rows:
        raise ValueError(f"Probability count {len(probabilities)} does not match matrix rows {matrix.rows}")

    sort_idx = np.argsort(-probabilities, kind="stable")
    probs_sorted = probabilities[sort_idx]
    s1_sorted = s1_indices[sort_idx]
    is_true_sorted = is_true_arr[sort_idx]

    pc_arr = np.zeros(entity_count, dtype=np.int32)
    tp_arr = np.zeros(entity_count, dtype=np.int32)
    f05_arr = np.where(tc_arr == 0, 1.0, 0.0)
    total_f05 = float(np.sum(f05_arr, dtype=np.float64))
    best_score = total_f05 / entity_count
    best_thresh = 1.0

    i = 0
    n_preds = len(probs_sorted)
    while i < n_preds:
        thresh = float(probs_sorted[i])
        while i < n_preds and float(probs_sorted[i]) == thresh:
            s1_idx = int(s1_sorted[i])
            total_f05 -= float(f05_arr[s1_idx])
            pc_arr[s1_idx] += 1
            if is_true_sorted[i]:
                tp_arr[s1_idx] += 1

            tc = int(tc_arr[s1_idx])
            pc = int(pc_arr[s1_idx])
            tp = int(tp_arr[s1_idx])
            if tc == 0:
                new_f05 = 0.0
            elif pc == 0 or tp == 0:
                new_f05 = 0.0
            else:
                precision = tp / pc
                recall = tp / tc
                denom = 0.25 * precision + recall
                new_f05 = (1.25 * precision * recall) / denom if denom else 0.0
            f05_arr[s1_idx] = new_f05
            total_f05 += new_f05
            i += 1

        score = total_f05 / entity_count
        if score > best_score or (score == best_score and thresh > best_thresh):
            best_score = float(score)
            best_thresh = thresh

    return float(best_score), float(best_thresh)


@dataclass
class ModelSelection:
    option: tuple[str, str | None, float | None]
    country_score: float
    country_threshold: float
    id_score: float
    id_threshold: float
    results: dict
    country_model: PairModel
    country_probs: np.ndarray
    id_model: PairModel
    id_probs: np.ndarray


def _model_options(labels: np.ndarray) -> list[tuple[str, str | None, float | None]]:
    positives = max(1, int(np.count_nonzero(labels)))
    imbalance = max(1.0, (len(labels) - positives) / positives)
    moderate_weight = min(imbalance, max(1.25, float(np.sqrt(imbalance))))
    return [
        ("unweighted", None, None),
        ("balanced", "balanced", None),
        (f"scale_pos_weight_{moderate_weight:.3f}", None, moderate_weight),
    ]


def _select_model(
    store: BlockingStore,
    fit: MatrixFiles,
    tune: MatrixFiles,
    in_fit: MatrixFiles,
    in_tune: MatrixFiles,
    where: str,
    parameters: Sequence[object],
    seed: int,
) -> ModelSelection:
    """Select model weights once and retain the winning validation artifacts.

    Retaining the winning models/probabilities avoids refitting the same model
    later just to report precision/recall or retune the in-distribution
    threshold.
    """
    results: dict = {}
    best_option = None
    best_score = -1.0
    best_country_threshold = 0.95
    best_country_model: PairModel | None = None
    best_country_probs: np.ndarray | None = None
    best_id_model: PairModel | None = None
    best_id_probs: np.ndarray | None = None
    best_id_score = -1.0
    best_id_threshold = 0.95
    baseline_id_score = None

    fit_y = fit.y()
    if fit_y is None:
        raise ValueError("Training matrix is missing labels")

    for option in _model_options(fit_y):
        name, class_weight, scale_pos_weight = option

        country_model = PairModel(
            seed,
            class_weight=class_weight,
            scale_pos_weight=scale_pos_weight,
        ).fit(fit.x(), fit_y)
        probs_tune = _batched_predict_proba(country_model, tune)
        country_score, country_threshold = _fast_tune_threshold(
            store, tune, probs_tune, where, parameters
        )

        id_y = in_fit.y()
        if id_y is None:
            raise ValueError("In-distribution training matrix is missing labels")
        id_model = PairModel(
            seed,
            class_weight=class_weight,
            scale_pos_weight=scale_pos_weight,
        ).fit(in_fit.x(), id_y)
        probs_id = _batched_predict_proba(id_model, in_tune)
        id_score, id_threshold = _fast_tune_threshold(
            store, in_tune, probs_id, "s.split=0", ()
        )

        if name == "unweighted":
            baseline_id_score = id_score

        results[name] = {
            "macro_f0_5": country_score,
            "threshold": country_threshold,
            "id_macro_f0_5": id_score,
            "id_threshold": id_threshold,
        }

        LOGGER.info(
            "Model %s: country F0.5=%.6f (t=%.3f), ID F0.5=%.6f (t=%.3f)",
            name,
            country_score,
            country_threshold,
            id_score,
            id_threshold,
        )

        if (
            baseline_id_score is not None
            and id_score < baseline_id_score - 0.005
        ):
            LOGGER.info(
                "Rejecting %s: ID score %.6f regressed from baseline %.6f",
                name,
                id_score,
                baseline_id_score,
            )
            continue

        is_better = (
            country_score > best_score
            or (
                country_score == best_score
                and country_threshold > best_country_threshold
            )
        )
        if is_better:
            best_option = option
            best_score = float(country_score)
            best_country_threshold = float(country_threshold)
            best_id_score = float(id_score)
            best_id_threshold = float(id_threshold)
            best_country_model = country_model
            best_country_probs = probs_tune
            best_id_model = id_model
            best_id_probs = probs_id

    if (
        best_option is None
        or best_country_model is None
        or best_country_probs is None
        or best_id_model is None
        or best_id_probs is None
    ):
        raise RuntimeError("No model option survived model selection")

    return ModelSelection(
        option=best_option,
        country_score=best_score,
        country_threshold=best_country_threshold,
        id_score=best_id_score,
        id_threshold=best_id_threshold,
        results=results,
        country_model=best_country_model,
        country_probs=best_country_probs,
        id_model=best_id_model,
        id_probs=best_id_probs,
    )

def validate(
    data_dir: str | Path,
    top_k: int = DEFAULT_TOP_K,
    seed: int = 42,
    scratch_dir: str | Path = "scratch",
) -> tuple[float, float]:
    t_total = time.perf_counter()
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    store = _build_store(
        data_dir,
        "train",
        scratch / "train_validation.sqlite",
        top_k,
        with_truth=True,
    )
    try:
        train_country, validation_country = _countries(store)

        fit = _materialize(
            store,
            scratch,
            "country_fit",
            "s.country=?",
            (train_country,),
            subsample=True,
        )
        valid = _materialize(
            store,
            scratch,
            "country_valid",
            "s.country=?",
            (validation_country,),
        )
        in_fit = _materialize(
            store,
            scratch,
            "id_fit",
            "s.split<>0",
            (),
            subsample=True,
        )
        in_valid = _materialize(
            store,
            scratch,
            "id_valid",
            "s.split=0",
            (),
        )

        selection = _select_model(
            store,
            fit,
            valid,
            in_fit,
            in_valid,
            "s.country=?",
            (validation_country,),
            seed,
        )

        # Reverse-country validation is a robustness check. Keep its training
        # side sampled so validation does not require another full candidate
        # matrix/model fit on the opposite country.
        rev_fit = _materialize(
            store,
            scratch,
            "country_rev_fit",
            "s.country=?",
            (validation_country,),
            subsample=True,
        )
        rev_valid = _materialize(
            store,
            scratch,
            "country_rev_valid",
            "s.country=?",
            (train_country,),
        )

        _, rev_cw, rev_spw = selection.option
        rev_model = PairModel(
            seed,
            class_weight=rev_cw,
            scale_pos_weight=rev_spw,
        ).fit(rev_fit.x(), rev_fit.y())
        rev_probs = _batched_predict_proba(rev_model, rev_valid)
        rev_score, rev_threshold = _fast_tune_threshold(
            store,
            rev_valid,
            rev_probs,
            "s.country=?",
            (train_country,),
        )

        # Reuse the selected in-distribution model/probabilities instead of
        # fitting the same model a second time just to compute the final ID
        # diagnostics.
        in_probs = selection.id_probs
        in_score = selection.id_score
        in_threshold = selection.id_threshold

        def _precision_recall(
            probs: np.ndarray,
            matrix: MatrixFiles,
            threshold: float,
        ) -> tuple[float, float]:
            labels = matrix.y()
            if labels is None:
                return 0.0, 0.0
            labels_bool = np.asarray(labels, dtype=bool)
            mask = probs >= threshold
            tp = int(np.sum(mask & labels_bool))
            fp = int(np.sum(mask & ~labels_bool))
            fn = int(np.sum(~mask & labels_bool))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            return precision, recall

        country_prec, country_rec = _precision_recall(
            selection.country_probs,
            valid,
            selection.country_threshold,
        )
        in_prec, in_rec = _precision_recall(
            in_probs,
            in_valid,
            in_threshold,
        )

        candidate_total, candidate_avg = store.candidate_summary()
        report = {
            "country_holdout": {
                "train_country": train_country,
                "validation_country": validation_country,
                "macro_f0_5": selection.country_score,
                "threshold": selection.country_threshold,
                "precision": country_prec,
                "recall": country_rec,
            },
            "reverse_country_holdout": {
                "train_country": validation_country,
                "validation_country": train_country,
                "macro_f0_5": rev_score,
                "threshold": rev_threshold,
            },
            "in_distribution": {
                "macro_f0_5": in_score,
                "threshold": in_threshold,
                "precision": in_prec,
                "recall": in_rec,
            },
            "model_selection": {
                "selected": selection.option[0],
                "ablation": selection.results,
            },
            "blocking": {
                "recall": store.recall_ceiling()[0],
                "candidate_pairs": candidate_total,
                "average_candidates": candidate_avg,
                "diagnostics": store.diagnostics,
            },
            "runtime_seconds": time.perf_counter() - t_total,
        }
        (scratch / "validation_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        LOGGER.info(
            "Validation complete: Country OOD F0.5=%.6f, ID F0.5=%.6f [%.1fs]",
            selection.country_score,
            in_score,
            time.perf_counter() - t_total,
        )
        LOGGER.info(
            "  Country: prec=%.4f rec=%.4f  |  ID: prec=%.4f rec=%.4f",
            country_prec,
            country_rec,
            in_prec,
            in_rec,
        )

        return selection.country_score, selection.country_threshold
    finally:
        store.close()

def predict(
    test_dir: str | Path,
    output_dir: str | Path,
    train_dir: str | Path,
    top_k: int = DEFAULT_TOP_K,
    seed: int = 42,
    scratch_dir: str | Path = "scratch",
) -> float:
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    train_store = _build_store(
        train_dir,
        "train",
        scratch / "train_predict.sqlite",
        top_k,
        with_truth=True,
    )

    try:
        train_country, validation_country = _countries(train_store)

        fit = _materialize(
            train_store,
            scratch,
            "threshold_fit",
            "s.country=?",
            (train_country,),
            subsample=True,
        )
        tune = _materialize(
            train_store,
            scratch,
            "threshold_tune",
            "s.country=?",
            (validation_country,),
        )
        in_fit = _materialize(
            train_store,
            scratch,
            "predict_id_fit",
            "s.split<>0",
            (),
            subsample=True,
        )
        in_tune = _materialize(
            train_store,
            scratch,
            "predict_id_tune",
            "s.split=0",
            (),
        )

        selection = _select_model(
            train_store,
            fit,
            tune,
            in_fit,
            in_tune,
            "s.country=?",
            (validation_country,),
            seed,
        )

        # Calibrate the production threshold from the retained winning
        # model's grouped in-distribution validation predictions.
        threshold = selection.id_threshold
        LOGGER.info(
            "Calibrated final threshold %.6f using S1-grouped validation (split=0)",
            threshold,
        )

        full = _materialize(
            train_store,
            scratch,
            "full_train",
            labels=True,
            subsample=True,
        )

        _, class_weight, scale_pos_weight = selection.option
        final_model = PairModel(
            seed,
            class_weight=class_weight,
            scale_pos_weight=scale_pos_weight,
        ).fit(full.x(), full.y())
    finally:
        train_store.close()

    test_store = _build_store(
        test_dir,
        "test",
        scratch / "test_predict.sqlite",
        top_k,
        with_truth=False,
    )
    try:
        # Stream final candidate features -> model -> outputs. This keeps peak
        # test-inference memory bounded by batch_size rather than all pairs.
        write_submission_from_store_streaming(
            store=test_store,
            model=final_model,
            output_dir=Path(output_dir),
            threshold=threshold,
        )
    finally:
        test_store.close()

    return threshold