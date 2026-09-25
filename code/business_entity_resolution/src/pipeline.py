"""Canonical full-data train, validate, and predict workflow."""
from __future__ import annotations

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
from .model import PairModel
from .output import write_submission_from_database

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
    LOGGER.info("Source-1 index: %d records (%.1fs)", source_count, time.perf_counter() - t_start)
    if with_truth:
        store.add_truth(Path(data_dir) / "train_ground_truth.tsv")
    t_ret = time.perf_counter()
    store.add_targets_and_retrieve((source2, source3))
    LOGGER.info("Target retrieval: %.1fs", time.perf_counter() - t_ret)
    pair_count = store.finalize_candidates()
    candidates, average = store.candidate_summary()
    LOGGER.info("Blocking complete: %d Source-1 records, %d final pairs, %.3f candidates/entity (%.1fs total)",
                source_count, candidates, average, time.perf_counter() - t_start)
    if with_truth:
        recall, retrieved, total = store.recall_ceiling()
        LOGGER.info("Blocking Recall Ceiling: %.4f%% (%d/%d)", 100 * recall, retrieved, total)
    return store


def _materialize(store: BlockingStore, scratch: Path, name: str, where: str = "", parameters: Sequence[object] = (), labels: bool = True) -> MatrixFiles:
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
    if written != rows:
        raise RuntimeError(f"Feature materialization changed from {rows} to {written} rows")
    return MatrixFiles(x_path, y_path, pairs_path, rows)


def _predict_to_store(model: PairModel, matrix: MatrixFiles, store: BlockingStore) -> None:
    with matrix.pairs_path.open("r", encoding="utf-8") as pair_file:
        x = matrix.x()
        offset = 0
        while offset < matrix.rows:
            end = min(matrix.rows, offset + 100_000)
            probabilities = model.predict_proba(x[offset:end])
            pairs = [tuple(line.rstrip("\n").split("\t", 1)) for line in [pair_file.readline() for _ in range(end - offset)]]
            store.update_probabilities(pairs, probabilities)
            offset = end


def _countries(store: BlockingStore) -> tuple[str, str]:
    values = list(store.connection.execute("SELECT country, COUNT(*) FROM source1 WHERE country<>'' GROUP BY country ORDER BY COUNT(*) DESC, country"))
    if len(values) < 2:
        raise ValueError("Country holdout requires at least two countries")
    return values[0][0], values[1][0]


def _macro_f0_5_sql(store: BlockingStore, threshold: float, where: str, parameters: Sequence[object]) -> float:
    """Exact macro F0.5 in SQL so validation never loads millions of labels."""
    query = f"""
        WITH scope AS (SELECT entity_id FROM source1 s WHERE {where}),
        truth_counts AS (SELECT t.source1_id, COUNT(*) AS n FROM truth t JOIN scope q ON q.entity_id=t.source1_id GROUP BY t.source1_id),
        prediction_counts AS (SELECT c.source1_id, COUNT(*) AS n FROM final_candidates c JOIN scope q ON q.entity_id=c.source1_id WHERE c.probability >= ? GROUP BY c.source1_id),
        true_positive_counts AS (
            SELECT c.source1_id, COUNT(*) AS n FROM final_candidates c JOIN truth t ON t.source1_id=c.source1_id AND t.target_id=c.target_id
            JOIN scope q ON q.entity_id=c.source1_id WHERE c.probability >= ? GROUP BY c.source1_id
        )
        SELECT AVG(CASE
            WHEN COALESCE(tc.n, 0) = 0 THEN CASE WHEN COALESCE(pc.n, 0) = 0 THEN 1.0 ELSE 0.0 END
            WHEN COALESCE(pc.n, 0) = 0 OR COALESCE(tp.n, 0) = 0 THEN 0.0
            ELSE (1.25 * (CAST(tp.n AS REAL) / pc.n) * (CAST(tp.n AS REAL) / tc.n)) /
                 (0.25 * (CAST(tp.n AS REAL) / pc.n) + (CAST(tp.n AS REAL) / tc.n))
        END) FROM scope q LEFT JOIN truth_counts tc ON tc.source1_id=q.entity_id
        LEFT JOIN prediction_counts pc ON pc.source1_id=q.entity_id LEFT JOIN true_positive_counts tp ON tp.source1_id=q.entity_id
    """
    value = store.connection.execute(query, (*parameters, threshold, threshold)).fetchone()[0]
    return float(value or 0.0)


def _tune_threshold(store: BlockingStore, where: str, parameters: Sequence[object]) -> tuple[float, float]:
    best_score, best_threshold = -1.0, 0.995
    thresholds = np.unique(np.concatenate((
        np.round(np.arange(0.30, 0.901, 0.01), 3),
        np.round(np.arange(0.901, 0.996, 0.001), 3),
    )))
    for threshold in thresholds:
        score = _macro_f0_5_sql(store, float(threshold), where, parameters)
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_score, best_threshold = score, float(threshold)
    return best_score, best_threshold


def _model_options(labels: np.ndarray) -> list[tuple[str, str | None, float | None]]:
    """A compact, validation-selected class-weight ablation.

    The final option is deliberately between 1 and the observed candidate-pair
    imbalance; using its square root avoids blindly applying the full ratio to a
    precision-weighted F0.5 objective.
    """
    positives = max(1, int(np.count_nonzero(labels)))
    imbalance = max(1.0, (len(labels) - positives) / positives)
    moderate_weight = min(imbalance, max(1.25, float(np.sqrt(imbalance))))
    return [
        ("unweighted", None, None),
        ("balanced", "balanced", None),
        (f"scale_pos_weight_{moderate_weight:.3f}", None, moderate_weight),
    ]


def _select_model(store: BlockingStore, fit: MatrixFiles, tune: MatrixFiles,
                  where: str, parameters: Sequence[object], seed: int) -> tuple[tuple[str, str | None, float | None], float, float, dict[str, dict[str, float]]]:
    """Choose weighting only by country-OOD macro F0.5, then restore its scores."""
    results: dict[str, dict[str, float]] = {}
    best: tuple[str, str | None, float | None] | None = None
    best_score, best_threshold = -1.0, 0.995
    for option in _model_options(fit.y()):  # type: ignore[arg-type]
        name, class_weight, scale_pos_weight = option
        model = PairModel(seed, class_weight=class_weight, scale_pos_weight=scale_pos_weight).fit(fit.x(), fit.y())
        _predict_to_store(model, tune, store)
        score, threshold = _tune_threshold(store, where, parameters)
        results[name] = {"macro_f0_5": score, "threshold": threshold}
        if score > best_score or (score == best_score and threshold > best_threshold):
            best, best_score, best_threshold = option, score, threshold
    assert best is not None
    # Later diagnostics and error exports should contain probabilities from the
    # chosen model, rather than the configuration evaluated last.
    _, class_weight, scale_pos_weight = best
    chosen = PairModel(seed, class_weight=class_weight, scale_pos_weight=scale_pos_weight).fit(fit.x(), fit.y())
    _predict_to_store(chosen, tune, store)
    return best, best_score, best_threshold, results


def _write_errors(store: BlockingStore, path: Path, threshold: float, where: str, parameters: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("error_type,source1_entity_id,candidate_entity_id,probability,source1_name,source1_address,candidate_name,candidate_address\n")
        query = f"""
            SELECT CASE WHEN truth.target_id IS NULL THEN 'false_positive' ELSE 'false_negative' END,
                   c.source1_id, c.target_id, COALESCE(c.probability, 0), s.name, s.address, t.name, t.address
            FROM final_candidates c JOIN source1 s ON s.entity_id=c.source1_id JOIN targets t ON t.entity_id=c.target_id
            LEFT JOIN truth ON truth.source1_id=c.source1_id AND truth.target_id=c.target_id
            WHERE ((c.probability >= ? AND truth.target_id IS NULL) OR (c.probability < ? AND truth.target_id IS NOT NULL)) AND {where}
        """
        import csv
        writer = csv.writer(stream)
        writer.writerows(store.connection.execute(query, (threshold, threshold, *parameters)))


def validate(data_dir: str | Path, top_k: int = 30, seed: int = 42, scratch_dir: str | Path = "scratch") -> tuple[float, float]:
    """Run country-OOD and singleton-stratified checks on the real training data."""
    t_total = time.perf_counter()
    scratch = Path(scratch_dir); scratch.mkdir(parents=True, exist_ok=True)
    store = _build_store(data_dir, "train", scratch / "train_validation.sqlite", top_k, with_truth=True)
    train_country, validation_country = _countries(store)
    LOGGER.info("Country holdout: train=%s, validation=%s", train_country, validation_country)

    # ── Country OOD: train→validation ──
    t0 = time.perf_counter()
    fit = _materialize(store, scratch, "country_fit", "s.country=?", (train_country,))
    valid = _materialize(store, scratch, "country_valid", "s.country=?", (validation_country,))
    LOGGER.info("Feature materialization: %.1fs", time.perf_counter() - t0)
    selected, country_score, country_threshold, weight_results = _select_model(
        store, fit, valid, "s.country=?", (validation_country,), seed,
    )
    _write_errors(store, scratch / "country_holdout_errors.csv", country_threshold, "s.country=?", (validation_country,))

    # ── Country OOD: reverse direction (validation→train) ──
    rev_fit = _materialize(store, scratch, "country_rev_fit", "s.country=?", (validation_country,))
    rev_valid = _materialize(store, scratch, "country_rev_valid", "s.country=?", (train_country,))
    _, rev_class_weight, rev_scale_pos_weight = selected
    rev_model = PairModel(seed, class_weight=rev_class_weight, scale_pos_weight=rev_scale_pos_weight).fit(rev_fit.x(), rev_fit.y())
    _predict_to_store(rev_model, rev_valid, store)
    rev_score, rev_threshold = _tune_threshold(store, "s.country=?", (train_country,))
    LOGGER.info("Reverse OOD (%s→%s): macro F0.5=%.6f (threshold=%.2f)",
                validation_country, train_country, rev_score, rev_threshold)

    # ── In-distribution split ──
    in_fit = _materialize(store, scratch, "id_fit", "s.split<>0", ())
    in_valid = _materialize(store, scratch, "id_valid", "s.split=0", ())
    _, class_weight, scale_pos_weight = selected
    in_model = PairModel(seed, class_weight=class_weight, scale_pos_weight=scale_pos_weight).fit(in_fit.x(), in_fit.y())
    _predict_to_store(in_model, in_valid, store)
    in_score, in_threshold = _tune_threshold(store, "s.split=0", ())

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
        "model_selection": {"selected": selected[0], "country_holdout_ablation": weight_results},
        "blocking": {
            "recall": store.recall_ceiling()[0],
            "candidate_pairs": store.candidate_summary()[0],
            "average_candidates": store.candidate_summary()[1],
            "diagnostics": store.diagnostics,
        },
        "runtime_seconds": time.perf_counter() - t_total,
    }
    (scratch / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Country OOD macro F0.5=%.6f (threshold=%.2f); reverse=%.6f; in-distribution=%.6f (threshold=%.2f)  [%.0fs]",
                country_score, country_threshold, rev_score, in_score, in_threshold, time.perf_counter() - t_total)
    store.close()
    return country_score, country_threshold


def predict(test_dir: str | Path, output_dir: str | Path, train_dir: str | Path, top_k: int = 30, seed: int = 42, scratch_dir: str | Path = "scratch") -> float:
    """Build, train, score, and serialize the only supported production pipeline."""
    scratch = Path(scratch_dir); scratch.mkdir(parents=True, exist_ok=True)
    train_store = _build_store(train_dir, "train", scratch / "train_predict.sqlite", top_k, with_truth=True)
    train_country, validation_country = _countries(train_store)
    fit = _materialize(train_store, scratch, "threshold_fit", "s.country=?", (train_country,))
    tune = _materialize(train_store, scratch, "threshold_tune", "s.country=?", (validation_country,))
    selected, country_score, country_threshold, _ = _select_model(
        train_store, fit, tune, "s.country=?", (validation_country,), seed,
    )
    in_fit = _materialize(train_store, scratch, "predict_id_fit", "s.split<>0", ())
    in_tune = _materialize(train_store, scratch, "predict_id_tune", "s.split=0", ())
    _, class_weight, scale_pos_weight = selected
    in_model = PairModel(seed, class_weight=class_weight, scale_pos_weight=scale_pos_weight).fit(in_fit.x(), in_fit.y())
    _predict_to_store(in_model, in_tune, train_store)
    in_score, in_threshold = _tune_threshold(train_store, "s.split=0", ())
    # The country holdout is the OOD proxy for France.  Taking the more
    # conservative of it and the independent in-distribution threshold retains
    # that signal while protecting the precision-heavy singleton metric.
    threshold = max(country_threshold, in_threshold)
    LOGGER.info("Threshold policy: max(country-OOD %.3f, in-distribution %.3f) = %.3f; scores %.6f / %.6f",
                country_threshold, in_threshold, threshold, country_score, in_score)
    full = _materialize(train_store, scratch, "full_train", labels=True)
    model = PairModel(seed, class_weight=class_weight, scale_pos_weight=scale_pos_weight).fit(full.x(), full.y())
    train_store.close()

    test_store = _build_store(test_dir, "test", scratch / "test_predict.sqlite", top_k, with_truth=False)
    test = _materialize(test_store, scratch, "test", labels=False)
    _predict_to_store(model, test, test_store)
    write_submission_from_database(output_dir, test_store.path, threshold)
    test_store.close()
    LOGGER.info("Prediction complete with threshold %.2f", threshold)
    return threshold
