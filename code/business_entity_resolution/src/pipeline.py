"""Canonical train, validate, and predict workflow with fast threshold search and streaming inference."""
from __future__ import annotations

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
from .output import write_submission_stream

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatrixFiles:
    x_path: Path
    y_path: Path | None
    pairs_path: Path
    rows: int

    def x(self, mode: str = "r") -> np.memmap:
        return np.memmap(self.x_path, dtype=np.float32, mode=mode, shape=(self.rows, len(FEATURE_NAMES)))

    def y(self) -> np.memmap | None:
        return None if self.y_path is None else np.memmap(self.y_path, dtype=np.int8, mode="r", shape=(self.rows,))


def _dataset_paths(directory: str | Path, split: str) -> tuple[Path, Path, Path]:
    root = Path(directory)
    return tuple(root / f"{split}_source{source}.tsv" for source in (1, 2, 3))  # type: ignore[return-value]


def _build_store(data_dir: str | Path, split: str, database: Path, top_k: int, with_truth: bool) -> BlockingStore:
    t_start = time.perf_counter()
    source1, source2, source3 = _dataset_paths(data_dir, split)
    store = BlockingStore(database, top_k=top_k)
    store.reset()
    source_count = store.build_source_index(source1)
    if with_truth:
        store.add_truth(Path(data_dir) / "train_ground_truth.tsv")
    store.add_targets_and_retrieve((source2, source3))
    store.finalize_candidates()
    candidates, average = store.candidate_summary()
    LOGGER.info("Blocking complete: %d Source-1 records, %d pairs (%.3f/entity) in %.1fs",
                source_count, candidates, average, time.perf_counter() - t_start)
    return store


def _materialize(store: BlockingStore, scratch: Path, name: str, where: str = "", parameters: Sequence[object] = (), labels: bool = True, subsample: bool = False) -> MatrixFiles:
    if labels and subsample:
        cond = "(truth.target_id IS NOT NULL OR c.similarity > 0.65 OR c.rank <= 4 OR (length(s.name) + length(t.name) + c.rank) % 10 = 0)"
        where = f"({where}) AND {cond}" if where else cond
    rows = store.feature_count(where, parameters)
    x_path, pairs_path = scratch / f"{name}.features.f32", scratch / f"{name}.pairs.tsv"
    y_path = scratch / f"{name}.labels.i8" if labels else None
    x = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(rows, len(FEATURE_NAMES)))
    y = None if y_path is None else np.memmap(y_path, dtype=np.int8, mode="w+", shape=(rows,))
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
    """Exact O(N log N) threshold optimization over all observed probabilities."""
    s1_rows = store.connection.execute(f"SELECT entity_id FROM source1 s WHERE {where} ORDER BY entity_id", parameters).fetchall()
    all_s1_list = [r[0] for r in s1_rows]
    total_entities = len(all_s1_list)
    s1_to_idx = {sid: i for i, sid in enumerate(all_s1_list)}

    tc_arr = np.zeros(total_entities, dtype=np.int32)
    truth_rows = store.connection.execute(
        f"SELECT t.source1_id, COUNT(*) FROM truth t JOIN source1 s ON s.entity_id=t.source1_id WHERE {where} GROUP BY t.source1_id",
        parameters
    ).fetchall()
    for sid, count in truth_rows:
        if sid in s1_to_idx:
            tc_arr[s1_to_idx[sid]] = count

    s1_indices = []
    with matrix.pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            s1_indices.append(s1_to_idx[line.split("\t", 1)[0]])
    s1_indices_arr = np.asarray(s1_indices, dtype=np.int32)

    labels = matrix.y()
    is_true_arr = np.asarray(labels, dtype=bool) if labels is not None else np.zeros(len(s1_indices_arr), dtype=bool)

    sort_idx = np.argsort(-probabilities)
    probs_sorted = probabilities[sort_idx]
    s1_sorted = s1_indices_arr[sort_idx]
    is_true_sorted = is_true_arr[sort_idx]

    pc_arr = np.zeros(total_entities, dtype=np.int32)
    tp_arr = np.zeros(total_entities, dtype=np.int32)
    
    # Initialize f05_arr
    # If tc_arr[i] == 0, initially pc == 0, so f05 is 1.0. If tc_arr > 0, initially f05 is 0.0.
    f05_arr = np.where(tc_arr == 0, 1.0, 0.0)
    total_f05 = np.sum(f05_arr)

    best_score = float(total_f05 / total_entities) if total_entities > 0 else 0.0
    best_thresh = 1.0

    n_preds = len(probs_sorted)
    i = 0
    while i < n_preds:
        thresh = probs_sorted[i]
        
        while i < n_preds and probs_sorted[i] == thresh:
            s1 = s1_sorted[i]
            is_tp = is_true_sorted[i]
            
            total_f05 -= f05_arr[s1]
            
            pc_arr[s1] += 1
            if is_tp:
                tp_arr[s1] += 1
                
            tc = tc_arr[s1]
            pc = pc_arr[s1]
            tp = tp_arr[s1]
            
            if tc == 0:
                new_f05 = 1.0 if pc == 0 else 0.0
            elif pc == 0 or tp == 0:
                new_f05 = 0.0
            else:
                p = tp / pc
                r = tp / tc
                denom = 0.25 * p + r
                new_f05 = (1.25 * p * r) / denom if denom > 0 else 0.0
                
            f05_arr[s1] = new_f05
            total_f05 += new_f05
            
            i += 1
            
        score = float(total_f05 / total_entities)
        if score > best_score or (score == best_score and float(thresh) > best_thresh):
            best_score = score
            best_thresh = float(thresh)

    return best_score, best_thresh


def _model_options(labels: np.ndarray) -> list[tuple[str, str | None, float | None]]:
    positives = max(1, int(np.count_nonzero(labels)))
    imbalance = max(1.0, (len(labels) - positives) / positives)
    moderate_weight = min(imbalance, max(1.25, float(np.sqrt(imbalance))))
    return [
        ("unweighted", None, None),
        ("balanced", "balanced", None),
        (f"scale_pos_weight_{moderate_weight:.3f}", None, moderate_weight),
    ]


def _select_model(store: BlockingStore, fit: MatrixFiles, tune: MatrixFiles,
                  in_fit: MatrixFiles, in_tune: MatrixFiles,
                  where: str, parameters: Sequence[object], seed: int) -> tuple[tuple[str, str | None, float | None], float, float, dict]:
    results = {}
    best = None
    best_score, best_threshold = -1.0, 0.95
    baseline_id_score = None

    for option in _model_options(fit.y()):  # type: ignore[arg-type]
        name, class_weight, scale_pos_weight = option
        model = PairModel(seed, class_weight=class_weight, scale_pos_weight=scale_pos_weight).fit(fit.x(), fit.y())
        probs_tune = _batched_predict_proba(model, tune)
        score, threshold = _fast_tune_threshold(store, tune, probs_tune, where, parameters)

        in_model = PairModel(seed, class_weight=class_weight, scale_pos_weight=scale_pos_weight).fit(in_fit.x(), in_fit.y())
        probs_id = _batched_predict_proba(in_model, in_tune)
        id_score, _ = _fast_tune_threshold(store, in_tune, probs_id, "s.split=0", ())

        if name == "unweighted":
            baseline_id_score = id_score

        results[name] = {"macro_f0_5": score, "threshold": threshold, "id_macro_f0_5": id_score}

        if baseline_id_score is not None and id_score < baseline_id_score - 0.005:
            LOGGER.info("Rejecting %s: ID score %.6f regressed from baseline %.6f", name, id_score, baseline_id_score)
            continue

        if score > best_score or (score == best_score and threshold > best_threshold):
            best, best_score, best_threshold = option, score, threshold

    assert best is not None
    return best, best_score, best_threshold, results


def validate(data_dir: str | Path, top_k: int = 30, seed: int = 42, scratch_dir: str | Path = "scratch") -> tuple[float, float]:
    t_total = time.perf_counter()
    scratch = Path(scratch_dir); scratch.mkdir(parents=True, exist_ok=True)
    store = _build_store(data_dir, "train", scratch / "train_validation.sqlite", top_k, with_truth=True)
    train_country, validation_country = _countries(store)

    fit = _materialize(store, scratch, "country_fit", "s.country=?", (train_country,), subsample=True)
    valid = _materialize(store, scratch, "country_valid", "s.country=?", (validation_country,))
    in_fit = _materialize(store, scratch, "id_fit", "s.split<>0", (), subsample=True)
    in_valid = _materialize(store, scratch, "id_valid", "s.split=0", ())

    selected, country_score, country_threshold, weight_results = _select_model(
        store, fit, valid, in_fit, in_valid, "s.country=?", (validation_country,), seed,
    )

    rev_fit = _materialize(store, scratch, "country_rev_fit", "s.country=?", (validation_country,))
    rev_valid = _materialize(store, scratch, "country_rev_valid", "s.country=?", (train_country,))
    _, rev_cw, rev_spw = selected
    rev_model = PairModel(seed, class_weight=rev_cw, scale_pos_weight=rev_spw).fit(rev_fit.x(), rev_fit.y())
    rev_probs = _batched_predict_proba(rev_model, rev_valid)
    rev_score, rev_threshold = _fast_tune_threshold(store, rev_valid, rev_probs, "s.country=?", (train_country,))

    in_model = PairModel(seed, class_weight=rev_cw, scale_pos_weight=rev_spw).fit(in_fit.x(), in_fit.y())
    in_probs = _batched_predict_proba(in_model, in_valid)
    in_score, in_threshold = _fast_tune_threshold(store, in_valid, in_probs, "s.split=0", ())

    report = {
        "country_holdout": {
            "train_country": train_country, "validation_country": validation_country,
            "macro_f0_5": country_score, "threshold": country_threshold,
        },
        "reverse_country_holdout": {
            "train_country": validation_country, "validation_country": train_country,
            "macro_f0_5": rev_score, "threshold": rev_threshold,
        },
        "in_distribution": {"macro_f0_5": in_score, "threshold": in_threshold},
        "model_selection": {"selected": selected[0], "ablation": weight_results},
        "blocking": {
            "recall": store.recall_ceiling()[0],
            "candidate_pairs": store.candidate_summary()[0],
            "average_candidates": store.candidate_summary()[1],
            "diagnostics": store.diagnostics,
        },
        "runtime_seconds": time.perf_counter() - t_total,
    }
    (scratch / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Validation complete: Country OOD F0.5=%.6f, ID F0.5=%.6f [%.1fs]",
                country_score, in_score, time.perf_counter() - t_total)
    store.close()
    return country_score, country_threshold


def predict(test_dir: str | Path, output_dir: str | Path, train_dir: str | Path, top_k: int = 30, seed: int = 42, scratch_dir: str | Path = "scratch") -> float:
    scratch = Path(scratch_dir); scratch.mkdir(parents=True, exist_ok=True)
    train_store = _build_store(train_dir, "train", scratch / "train_predict.sqlite", top_k, with_truth=True)
    train_country, validation_country = _countries(train_store)

    fit = _materialize(train_store, scratch, "threshold_fit", "s.country=?", (train_country,), subsample=True)
    tune = _materialize(train_store, scratch, "threshold_tune", "s.country=?", (validation_country,))
    in_fit = _materialize(train_store, scratch, "predict_id_fit", "s.split<>0", (), subsample=True)
    in_tune = _materialize(train_store, scratch, "predict_id_tune", "s.split=0", ())

    selected, _, country_threshold, _ = _select_model(
        train_store, fit, tune, in_fit, in_tune, "s.country=?", (validation_country,), seed,
    )

    _, class_weight, scale_pos_weight = selected
    in_model = PairModel(seed, class_weight=class_weight, scale_pos_weight=scale_pos_weight).fit(in_fit.x(), in_fit.y())
    in_probs = _batched_predict_proba(in_model, in_tune)
    _, threshold = _fast_tune_threshold(train_store, in_tune, in_probs, "s.split=0", ())
    
    LOGGER.info("Calibrated final threshold %.3f using S1-grouped validation (split=0)", threshold)
    
    full = _materialize(train_store, scratch, "full_train", labels=True, subsample=True)
    model = PairModel(seed, class_weight=class_weight, scale_pos_weight=scale_pos_weight).fit(full.x(), full.y())
    train_store.close()

    test_store = _build_store(test_dir, "test", scratch / "test_predict.sqlite", top_k, with_truth=False)
    test = _materialize(test_store, scratch, "test", labels=False)

    test_source1 = _dataset_paths(test_dir, "test")[0]
    test_probs = _batched_predict_proba(model, test)
    
    write_submission_stream(
        test_source1_path=test_source1,
        pairs_path=test.pairs_path,
        probabilities=test_probs,
        output_dir=Path(output_dir),
        threshold=threshold,
    )

    test_store.close()
    return threshold