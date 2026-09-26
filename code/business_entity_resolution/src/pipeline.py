"""Reproducible baseline train/validate/predict workflow.

This version intentionally separates the required first real-data run from later
model/threshold tuning. The first validation run uses a deterministic unweighted
LightGBM model and a fixed 0.5 decision threshold, records the complete real
validation report, and only then permits prediction. Later tuning can use that
report as the evidence baseline instead of guessing before a real run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .blocking import BlockingStore
from .features import FEATURE_NAMES, feature_batch
from .model import LinearPairModel, PairModel, ProbabilityEnsemble
from .variation import VariationModel
from .output import (
    validate_output_against_store,
    write_submission_from_store_streaming,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_TOP_K = BlockingStore.TOP_K
BASELINE_THRESHOLD = 0.5
CACHE_SCHEMA_VERSION = "2026-09-26-entity-resolution-v10-relation-aware"


@dataclass(frozen=True)
class MatrixFiles:
    x_path: Path
    y_path: Path | None
    pairs_path: Path
    rows: int

    def x(self, mode: str = "r") -> np.ndarray | np.memmap:
        if self.rows == 0:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        return np.memmap(
            self.x_path,
            dtype=np.float32,
            mode=mode,
            shape=(self.rows, len(FEATURE_NAMES)),
        )

    def y(self) -> np.ndarray | np.memmap | None:
        if self.y_path is None:
            return None
        if self.rows == 0:
            return np.empty((0,), dtype=np.int8)
        return np.memmap(self.y_path, dtype=np.int8, mode="r", shape=(self.rows,))


def _module_signature(*names: str) -> dict[str, str]:
    result: dict[str, str] = {}
    base = Path(__file__).resolve().parent
    for name in names:
        path = base / name
        if path.exists():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:20]
    return result


def _assert_materialized(paths: Sequence[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required dataset file not found: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            first = handle.readline().strip()
        if first == "version https://git-lfs.github.com/spec/v1":
            raise RuntimeError(
                f"Dataset file {path} is a Git-LFS pointer, not materialized challenge data. "
                "Fetch the official challenge dataset before running the pipeline."
            )


def _dataset_paths(directory: str | Path, split: str) -> tuple[Path, Path, Path]:
    root = Path(directory)
    return tuple(root / f"{split}_source{i}.tsv" for i in (1, 2, 3))  # type: ignore[return-value]


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _store_signature(data_dir: str | Path, split: str, top_k: int, with_truth: bool) -> dict:
    root = Path(data_dir)
    source_paths = [root / f"{split}_source{i}.tsv" for i in (1, 2, 3)]
    return {
        "split": split,
        "top_k": int(top_k),
        "with_truth": bool(with_truth),
        "source_files": {p.name: _file_signature(p) for p in source_paths},
        "truth_file": _file_signature(root / "train_ground_truth.tsv")
        if with_truth and (root / "train_ground_truth.tsv").exists()
        else None,
        "blocking_config": {
            "minhash_permutations": BlockingStore.MINHASH_PERMUTATIONS,
            "minhash_bands": BlockingStore.MINHASH_BANDS,
            "token_df_limit": BlockingStore.TOKEN_DF_LIMIT,
            "key_freq_max_source": BlockingStore.KEY_FREQ_MAX_SOURCE,
            "key_freq_max_target": BlockingStore.KEY_FREQ_MAX_TARGET,
            "high_freq_rescue_top": BlockingStore.HIGH_FREQ_RESCUE_TOP,
            "high_freq_rescue_max_block": BlockingStore.HIGH_FREQ_RESCUE_MAX_BLOCK,
            "shortlist_multiplier": BlockingStore.SHORTLIST_MULTIPLIER,
            "min_final_candidates": BlockingStore.MIN_FINAL_CANDIDATES,
            "ambiguous_min_final_candidates": BlockingStore.AMBIGUOUS_MIN_FINAL_CANDIDATES,
            "final_score_floor": BlockingStore.FINAL_SCORE_FLOOR,
            "final_score_margin": BlockingStore.FINAL_SCORE_MARGIN,
        },
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "code_signatures": _module_signature(
            "blocking.py", "preprocessing.py", "features.py", "variation.py", "model.py", "pipeline.py"
        ),
    }


def _build_store(
    data_dir: str | Path,
    split: str,
    database: Path,
    top_k: int,
    with_truth: bool,
) -> BlockingStore:
    t_start = time.perf_counter()
    source1, source2, source3 = _dataset_paths(data_dir, split)
    expected = _store_signature(data_dir, split, top_k, with_truth)
    _assert_materialized(
        [source1, source2, source3]
        + ([Path(data_dir) / "train_ground_truth.tsv"] if with_truth else [])
    )
    database.parent.mkdir(parents=True, exist_ok=True)
    store = BlockingStore(database, top_k=top_k)

    try:
        row = store.connection.execute(
            "SELECT value FROM pipeline_meta WHERE key='signature'"
        ).fetchone()
        diag_row = store.connection.execute(
            "SELECT value FROM pipeline_meta WHERE key='diagnostics'"
        ).fetchone()
        existing = json.loads(row[0]) if row else None
        cached_diagnostics = json.loads(diag_row[0]) if diag_row else {}
        has_final = (
            store.connection.execute("SELECT COUNT(*) FROM final_candidates").fetchone()[0] > 0
        )
    except sqlite3.OperationalError:
        existing, cached_diagnostics, has_final = None, {}, False

    if has_final and existing == expected:
        store.diagnostics = cached_diagnostics
        store.diagnostics["cache_reused"] = True
        LOGGER.info("Reusing validated candidate store: %s", database)
        return store

    if has_final:
        LOGGER.info("Invalid/stale candidate cache at %s; rebuilding", database)

    store.reset()
    source_count = store.build_source_index(source1)
    if with_truth:
        store.add_truth(Path(data_dir) / "train_ground_truth.tsv")
    store.add_targets_and_retrieve((source2, source3))
    store.finalize_candidates()

    store.connection.execute(
        "CREATE TABLE IF NOT EXISTS pipeline_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    store.connection.execute(
        "INSERT OR REPLACE INTO pipeline_meta(key, value) VALUES ('signature', ?)",
        (json.dumps(expected, sort_keys=True),),
    )
    store.connection.execute(
        "INSERT OR REPLACE INTO pipeline_meta(key, value) VALUES ('diagnostics', ?)",
        (json.dumps(store.diagnostics, sort_keys=True),),
    )
    store.connection.commit()

    candidates, average = store.candidate_summary()
    LOGGER.info(
        "Blocking complete: %d Source-1 records, %d final pairs (%.3f/entity) in %.1fs",
        source_count,
        candidates,
        average,
        time.perf_counter() - t_start,
    )
    return store


def _materialize(
    store: BlockingStore,
    scratch: Path,
    name: str,
    where: str = "",
    parameters: Sequence[object] = (),
    labels: bool = True,
    subsample: bool = False,
    variation_model: VariationModel | None = None,
) -> MatrixFiles:
    if labels and subsample:
        cond = (
            "(truth.target_id IS NOT NULL OR c.similarity > 0.55 "
            "OR c.rank <= 4 "
            "OR (length(s.name) + length(t.name) + c.rank) % 10 = 0)"
        )
        where = f"({where}) AND {cond}" if where else cond

    rows = store.feature_count(where, parameters)
    x_path = scratch / f"{name}.features.f32"
    pairs_path = scratch / f"{name}.pairs.tsv"
    y_path = scratch / f"{name}.labels.i8" if labels else None

    scratch.mkdir(parents=True, exist_ok=True)
    if rows:
        x = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(rows, len(FEATURE_NAMES)))
        y = np.memmap(y_path, dtype=np.int8, mode="w+", shape=(rows,)) if y_path else None
    else:
        x_path.write_bytes(b"")
        if y_path:
            y_path.write_bytes(b"")
        x = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        y = np.empty((0,), dtype=np.int8) if labels else None

    written = 0
    cursor = store.feature_rows(where, parameters)
    with pairs_path.open("w", encoding="utf-8", newline="") as pair_file:
        while True:
            batch = []
            try:
                for _ in range(25_000):
                    batch.append(next(cursor))
            except StopIteration:
                pass
            if not batch:
                break
            x[written : written + len(batch)] = feature_batch(
                [row[:-1] for row in batch], variation_model=variation_model
            )
            if y is not None:
                y[written : written + len(batch)] = [row[-1] for row in batch]
            pair_file.writelines(f"{row[0]}\t{row[1]}\n" for row in batch)
            written += len(batch)

    x.flush()
    if y is not None and hasattr(y, "flush"):
        y.flush()
    if written != rows:
        raise AssertionError(f"Materialization wrote {written} rows but expected {rows}")
    LOGGER.info("Materialized %s: %d rows x %d features", name, rows, len(FEATURE_NAMES))
    return MatrixFiles(x_path, y_path, pairs_path, rows)


def _batched_predict_proba(model: PairModel, matrix: MatrixFiles, batch_size: int = 100_000) -> np.ndarray:
    x = matrix.x()
    probs = np.zeros(matrix.rows, dtype=np.float32)
    for start in range(0, matrix.rows, batch_size):
        end = min(matrix.rows, start + batch_size)
        probs[start:end] = model.predict_proba(x[start:end])
    return probs


def _countries(store: BlockingStore) -> tuple[str, str]:
    values = list(
        store.connection.execute(
            "SELECT country, COUNT(*) FROM source1 WHERE country<>'' "
            "GROUP BY country ORDER BY COUNT(*) DESC, country"
        )
    )
    if len(values) < 2:
        raise ValueError("Country holdout requires at least two Source-1 countries")
    return values[0][0], values[1][0]


def _grouped_f05(
    store: BlockingStore,
    matrix: MatrixFiles,
    probabilities: np.ndarray,
    threshold: float,
    where: str,
    parameters: Sequence[object] = (),
) -> tuple[float, dict[str, float]]:
    """Compute exact macro F0.5 with the full truth denominator, including missed candidates."""
    labels = matrix.y()
    if labels is None:
        raise ValueError("Grouped F0.5 requires labels")

    s1_ids = [
        row[0]
        for row in store.connection.execute(
            f"SELECT s.entity_id FROM source1 s WHERE {where} ORDER BY s.entity_id", parameters
        )
    ]
    truth_counts = {
        sid: int(count)
        for sid, count in store.connection.execute(
            f"""SELECT t.source1_id, COUNT(*)
                 FROM truth t JOIN source1 s ON s.entity_id=t.source1_id
                 WHERE {where} GROUP BY t.source1_id""",
            parameters,
        )
    }
    pred_counts: dict[str, int] = {}
    tp_counts: dict[str, int] = {}

    pair_index = 0
    current_sid = None
    with matrix.pairs_path.open("r", encoding="utf-8") as pair_file:
        for line in pair_file:
            sid = line.split("\t", 1)[0]
            if current_sid != sid:
                current_sid = sid
            pred = bool(probabilities[pair_index] >= threshold)
            if pred:
                pred_counts[sid] = pred_counts.get(sid, 0) + 1
                if bool(labels[pair_index]):
                    tp_counts[sid] = tp_counts.get(sid, 0) + 1
            pair_index += 1
    if pair_index != matrix.rows:
        raise AssertionError("Pair/label stream length mismatch")

    scores: list[float] = []
    per_country: dict[str, list[float]] = {}
    countries = dict(store.connection.execute("SELECT entity_id,country FROM source1"))
    for sid in s1_ids:
        truth_count = truth_counts.get(sid, 0)
        pred_count = pred_counts.get(sid, 0)
        tp = tp_counts.get(sid, 0)
        if truth_count == 0:
            score = 1.0 if pred_count == 0 else 0.0
        elif pred_count == 0 or tp == 0:
            score = 0.0
        else:
            precision = tp / pred_count
            recall = tp / truth_count
            denom = 0.25 * precision + recall
            score = (1.25 * precision * recall) / denom if denom else 0.0
        score = float(score)
        scores.append(score)
        country = countries.get(sid, "") or "<missing>"
        per_country.setdefault(country, []).append(score)

    return (
        float(np.mean(scores)) if scores else 0.0,
        {country: float(np.mean(values)) for country, values in per_country.items()},
    )


def _pair_precision_recall(
    store: BlockingStore,
    matrix: MatrixFiles,
    probabilities: np.ndarray,
    threshold: float,
    where: str,
    parameters: Sequence[object] = (),
) -> tuple[float, float]:
    labels = matrix.y()
    if labels is None:
        raise ValueError("Pair precision/recall requires labels")
    truth_total = int(
        store.connection.execute(
            f"SELECT COUNT(*) FROM truth t JOIN source1 s ON s.entity_id=t.source1_id WHERE {where}",
            parameters,
        ).fetchone()[0]
    )
    mask = probabilities >= threshold
    truth = np.asarray(labels, dtype=bool)
    tp = int(np.sum(mask & truth))
    fp = int(np.sum(mask & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / truth_total if truth_total else 1.0
    return precision, recall


def _write_measured_documentation(report: dict) -> Path:
    """Write Documentation.md with the completed real-data baseline numbers."""
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "Documentation.md"
    blocking = report.get("blocking", {})
    raw = blocking.get("raw_blocking", {})
    final = blocking.get("final", {})
    stats = blocking.get("final_stats", {})
    rows = []
    for country, metrics in report.get("per_country_candidate_diagnostics", {}).items():
        rows.append(
            f"| {country} | {metrics.get('pair_recall', 0):.6f} | "
            f"{metrics.get('complete_match_set_recall', 0):.6f} | "
            f"{metrics.get('candidate_avg', 0):.2f} | {metrics.get('candidate_pairs', 0):,} |"
        )
    country_table = "\n".join(rows) or "| no country diagnostics | - | - | - | - |"
    text = f"""# Business Entity Resolution Challenge — Methodology and Measured Baseline\n\n## Objective\n\nThe pipeline resolves each Source-1 record to zero, one, or many Source-2/Source-3 records using only the supplied challenge data. The competition metric is macro F0.5 per Source-1 entity.\n\n## Preprocessing and blocking\n\nNames are Unicode-normalized, case-folded and legal-suffix canonicalized. Addresses are normalized with deterministic street/address aliases and conservative structured components. Blocking uses exact/near-exact keys, rare tokens, phonetic keys, address numbers/postal codes, and MinHash-LSH. Keys above the frequency ceiling are not truncated by arbitrary ID order.\n\nThe final candidate set is adaptive with a hard TOP_K ceiling and is exactly the set passed to the matching model.\n\n## Features\n\nThe baseline uses {report.get('feature_count', 0)} pair features covering exact/fuzzy name and address similarity, token similarity, acronym/legal-suffix signals, address components, country consistency, missingness, retrieval evidence, cross-field rank agreement, and reciprocal candidate rank.\n\n## Baseline model\n\nThe first required real-data run is intentionally untuned: unweighted LightGBM and a fixed decision threshold of 0.5. No class-weight ablation, ensemble, hysteresis tuning or pseudo-labeling is applied before this baseline report exists.\n\n## Measured real-data baseline\n\n- Overall final candidate pairs: **{report.get('candidate_pairs', 0):,}**\n- Average final candidates/S1: **{report.get('average_candidates_per_s1', 0.0):.2f}**\n- Raw pair recall: **{raw.get('recall', 0.0):.6f}**\n- Raw complete-match-set recall: **{raw.get('complete_match_set_recall', 0.0):.6f}**\n- Final pair recall: **{final.get('recall', 0.0):.6f}**\n- Final complete-match-set recall: **{final.get('complete_match_set_recall', 0.0):.6f}**\n- Final p95 candidates/S1: **{stats.get('p95', 0)}**\n- Final p99 candidates/S1: **{stats.get('p99', 0)}**\n- Runtime: **{report.get('runtime_seconds', 0.0):.1f} seconds**\n\n### Validation views\n\n| View | Macro F0.5 | Precision | Recall |\n|---|---:|---:|---:|\n| Country holdout | {report.get('country_holdout', {}).get('macro_f0_5', 0.0):.6f} | {report.get('country_holdout', {}).get('precision', 0.0):.6f} | {report.get('country_holdout', {}).get('recall', 0.0):.6f} |\n| Reverse country | {report.get('reverse_country_holdout', {}).get('macro_f0_5', 0.0):.6f} | {report.get('reverse_country_holdout', {}).get('precision', 0.0):.6f} | {report.get('reverse_country_holdout', {}).get('recall', 0.0):.6f} |\n| In-distribution | {report.get('in_distribution', {}).get('macro_f0_5', 0.0):.6f} | {report.get('in_distribution', {}).get('precision', 0.0):.6f} | {report.get('in_distribution', {}).get('recall', 0.0):.6f} |\n\n### Candidate diagnostics by Source-1 country\n\n| Country | Pair recall | Complete-set recall | Avg candidates/S1 | Candidate pairs |\n|---|---:|---:|---:|---:|\n{country_table}\n\n## Next stage\n\nThis completed real baseline is the evidence gate for subsequent tuning. Any new threshold, model weighting, ensemble, decision rule or pseudo-labeling strategy must be compared against these real held-out numbers before being retained.\n\n## Fair play\n\nNo external entity databases, APIs, geocoders, registries or web enrichment are used.\n"""
    path.write_text(text, encoding="utf-8")
    return path


def _build_variation_model(
    store: BlockingStore,
    where: str = "1=1",
    parameters: Sequence[object] = (),
    max_positive_pairs: int | None = None,
) -> VariationModel:
    """Build training-derived variation maps from a leakage-safe truth slice."""
    t0 = time.perf_counter()
    model = VariationModel.from_store(
        store, where=where, parameters=parameters, max_positive_pairs=max_positive_pairs
    )
    LOGGER.info(
        "Built variation model from %d positive pairs in %.1fs",
        model.positive_pairs,
        time.perf_counter() - t0,
    )
    return model


def _validate_policy_path(scratch: Path) -> Path:
    report_path = scratch / "validation_report.json"
    policy_path = scratch / "decision_policy.json"
    if not report_path.exists() or not policy_path.exists():
        raise RuntimeError(
            "Prediction is blocked until a completed real validation run exists. "
            "Run `python run.py --mode validate` first."
        )
    return policy_path


def validate(
    data_dir: str | Path,
    top_k: int = DEFAULT_TOP_K,
    seed: int = 42,
    scratch_dir: str | Path = "scratch",
) -> tuple[float, float]:
    """Run the required first real-data baseline; no hyperparameter search is performed."""
    t_total = time.perf_counter()
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    store = _build_store(
        data_dir,
        "train",
        scratch / "train.sqlite",
        top_k,
        with_truth=True,
    )
    try:
        train_country, validation_country = _countries(store)
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        country_fit = _materialize(
            store, scratch, "baseline_country_fit", "s.country=?", (train_country,), subsample=True
        )
        country_valid = _materialize(
            store, scratch, "baseline_country_valid", "s.country=?", (validation_country,)
        )
        timings["country_materialization_seconds"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        country_model = PairModel(seed).fit(country_fit.x(), country_fit.y())
        country_probs = _batched_predict_proba(country_model, country_valid)
        country_f05, country_f05_by_country = _grouped_f05(
            store, country_valid, country_probs, BASELINE_THRESHOLD, "s.country=?", (validation_country,)
        )
        country_prec, country_rec = _pair_precision_recall(
            store, country_valid, country_probs, BASELINE_THRESHOLD, "s.country=?", (validation_country,)
        )
        timings["country_model_seconds"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        reverse_fit = _materialize(
            store, scratch, "baseline_reverse_fit", "s.country=?", (validation_country,), subsample=True
        )
        reverse_valid = _materialize(
            store, scratch, "baseline_reverse_valid", "s.country=?", (train_country,)
        )
        reverse_model = PairModel(seed).fit(reverse_fit.x(), reverse_fit.y())
        reverse_probs = _batched_predict_proba(reverse_model, reverse_valid)
        reverse_f05, reverse_f05_by_country = _grouped_f05(
            store, reverse_valid, reverse_probs, BASELINE_THRESHOLD, "s.country=?", (train_country,)
        )
        reverse_prec, reverse_rec = _pair_precision_recall(
            store, reverse_valid, reverse_probs, BASELINE_THRESHOLD, "s.country=?", (train_country,)
        )
        timings["reverse_country_seconds"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        id_fit = _materialize(store, scratch, "baseline_id_fit", "s.split<>0", (), subsample=True)
        id_valid = _materialize(store, scratch, "baseline_id_valid", "s.split=0", ())
        id_model = PairModel(seed).fit(id_fit.x(), id_fit.y())
        id_probs = _batched_predict_proba(id_model, id_valid)
        id_f05, id_f05_by_country = _grouped_f05(
            store, id_valid, id_probs, BASELINE_THRESHOLD, "s.split=0", ()
        )
        id_prec, id_rec = _pair_precision_recall(
            store, id_valid, id_probs, BASELINE_THRESHOLD, "s.split=0", ()
        )
        timings["in_distribution_seconds"] = time.perf_counter() - t0

        candidate_total, candidate_avg = store.candidate_summary()
        policy = {
            "version": 1,
            "kind": "baseline_fixed_threshold",
            "default_threshold": BASELINE_THRESHOLD,
            "country_thresholds": {
                train_country: BASELINE_THRESHOLD,
                validation_country: BASELINE_THRESHOLD,
            },
            "zero_shot_fallback_threshold": BASELINE_THRESHOLD,
            "status": "baseline_not_tuned",
        }
        (scratch / "decision_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

        report = {
            "status": "completed",
            "baseline_only": True,
            "macro_f05_target": 0.98,
            "decision_policy": policy,
            "country_holdout": {
                "train_country": train_country,
                "validation_country": validation_country,
                "macro_f0_5": country_f05,
                "precision": country_prec,
                "recall": country_rec,
                "threshold": BASELINE_THRESHOLD,
                "macro_f0_5_by_country": country_f05_by_country,
            },
            "reverse_country_holdout": {
                "train_country": validation_country,
                "validation_country": train_country,
                "macro_f0_5": reverse_f05,
                "precision": reverse_prec,
                "recall": reverse_rec,
                "threshold": BASELINE_THRESHOLD,
                "macro_f0_5_by_country": reverse_f05_by_country,
            },
            "in_distribution": {
                "macro_f0_5": id_f05,
                "precision": id_prec,
                "recall": id_rec,
                "threshold": BASELINE_THRESHOLD,
                "macro_f0_5_by_country": id_f05_by_country,
            },
            "per_country_candidate_diagnostics": store.diagnostics.get("final", {}).get("by_country", {}),
            "blocking": store.diagnostics,
            "candidate_pairs": candidate_total,
            "average_candidates_per_s1": candidate_avg,
            "runtime_seconds": time.perf_counter() - t_total,
            "stage_runtime_seconds": timings,
            "feature_count": len(FEATURE_NAMES),
            "model": {
                "type": "LightGBM baseline",
                "class_weight": None,
                "scale_pos_weight": None,
                "tuning_performed": False,
            },
        }
        report_path = scratch / "validation_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        documentation_path = _write_measured_documentation(report)
        LOGGER.info("Wrote measured methodology document: %s", documentation_path)

        LOGGER.info(
            "Baseline validation complete: country F0.5=%.6f, reverse=%.6f, ID=%.6f [%.1fs]",
            country_f05,
            reverse_f05,
            id_f05,
            time.perf_counter() - t_total,
        )
        return country_f05, BASELINE_THRESHOLD
    finally:
        store.close()


def _production_model_from_policy(policy: dict, seed: int, X: np.ndarray, y: np.ndarray):
    spec = policy.get("selected_model_spec") or {}
    kind = str(spec.get("kind", "lgbm"))
    class_weight = spec.get("class_weight")
    scale_pos_weight = spec.get("scale_pos_weight")
    if kind == "ensemble":
        primary = PairModel(
            seed,
            class_weight=class_weight,
            scale_pos_weight=float(scale_pos_weight) if scale_pos_weight is not None else None,
        ).fit(X, y)
        secondary = LinearPairModel(seed, class_weight=class_weight).fit(X, y)
        return ProbabilityEnsemble(primary, secondary, primary_weight=float(spec.get("ensemble_weight", 0.75)))
    return PairModel(
        seed,
        class_weight=class_weight,
        scale_pos_weight=float(scale_pos_weight) if scale_pos_weight is not None else None,
    ).fit(X, y)


def predict(
    test_dir: str | Path,
    output_dir: str | Path,
    train_dir: str | Path,
    top_k: int = DEFAULT_TOP_K,
    seed: int = 42,
    scratch_dir: str | Path = "scratch",
) -> float:
    """Train on all training data and write both required submission files.

    Prediction uses the tuned policy when ``tuned_policy.json`` exists; otherwise
    it falls back to the measured baseline policy.  No prediction is permitted
    before a completed real validation baseline exists.
    """
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    baseline_policy_path = _validate_policy_path(scratch)
    baseline_policy = json.loads(baseline_policy_path.read_text(encoding="utf-8"))
    tuned_policy_path = scratch / "tuned_policy.json"
    policy = json.loads(tuned_policy_path.read_text(encoding="utf-8")) if tuned_policy_path.exists() else baseline_policy

    default_threshold = float(
        policy.get("id_threshold", policy.get("default_threshold", BASELINE_THRESHOLD))
    )
    threshold_by_country = {
        str(country).casefold(): float(value)
        for country, value in (policy.get("thresholds") or policy.get("country_thresholds") or {}).items()
    }
    unseen_threshold = float(
        policy.get("unseen_country_threshold", policy.get("zero_shot_fallback_threshold", default_threshold))
    )

    train_store = _build_store(
        train_dir,
        "train",
        scratch / "train.sqlite",
        top_k,
        with_truth=True,
    )
    variation_model = None
    try:
        spec = policy.get("selected_model_spec") or {}
        if bool(spec.get("use_variation", False)):
            variation_model = _build_variation_model(train_store, "1=1", (), None)
        full = _materialize(
            train_store,
            scratch,
            "production_full_train_var" if variation_model is not None else "production_full_train",
            labels=True,
            subsample=True,
            variation_model=variation_model,
        )
        labels = full.y()
        if labels is None:
            raise RuntimeError("Production training matrix has no labels")
        final_model = _production_model_from_policy(policy, seed, full.x(), labels)
    finally:
        train_store.close()

    test_store = _build_store(
        test_dir,
        "test",
        scratch / "test.sqlite",
        top_k,
        with_truth=False,
    )
    try:
        candidate_path, matching_path = write_submission_from_store_streaming(
            store=test_store,
            model=final_model,
            output_dir=Path(output_dir),
            threshold=default_threshold,
            threshold_by_country=threshold_by_country or None,
            unseen_country_threshold=unseen_threshold,
            variation_model=variation_model,
        )
        validate_output_against_store(
            store=test_store,
            candidate_path=candidate_path,
            matching_path=matching_path,
        )
    finally:
        test_store.close()

    return default_threshold
