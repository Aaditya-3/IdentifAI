"""Reproducible baseline train/validate/predict workflow.

This version intentionally separates the required first real-data run from later
model/threshold tuning. The first validation run uses a deterministic unweighted
LightGBM model and a fixed 0.5 decision threshold, records the complete real
validation report, and only then permits prediction. Later tuning can use that
report as the evidence baseline instead of guessing before a real run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .blocking import BlockingStore
from .features import FEATURE_NAMES, feature_batch
from .decision import DecisionPolicy
from .metrics import f05_from_counts
from .model import LinearPairModel, PairModel, ProbabilityEnsemble
from .semantic_retrieval import SemanticConfig
from .variation import VariationModel
from .output import (
    validate_output_against_store,
    write_submission_from_store_streaming,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_TOP_K = BlockingStore.TOP_K
BASELINE_THRESHOLD = 0.5
CACHE_SCHEMA_VERSION = "2026-09-27-entity-resolution-v13-bounded-semantic"
DEFAULT_MAX_TRAIN_ROWS = 2_000_000
DEFAULT_MIN_NEGATIVE_SAMPLE_MOD = 8
DEFAULT_MAX_NEGATIVE_SAMPLE_MOD = 512


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
    entity_columns = {"entity_id", "business_name", "business_address", "country"}
    truth_columns = {"source1_entity_id", "matched_entity_ids"}
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
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            fieldnames = set(csv.DictReader(handle, delimiter="\t").fieldnames or [])
        required = truth_columns if path.name == "train_ground_truth.tsv" else entity_columns
        missing = required.difference(fieldnames)
        if missing:
            expected = ", ".join(sorted(required))
            raise ValueError(
                f"Dataset file {path} has an invalid header. Missing columns: {', '.join(sorted(missing))}. "
                f"Expected columns: {expected}."
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
        "semantic_config": SemanticConfig.from_environment(top_k=top_k).signature(),
        "code_signatures": _module_signature(
            "blocking.py", "preprocessing.py", "features.py", "variation.py", "model.py", "decision.py", "semantic_retrieval.py", "tuning.py", "pipeline.py"
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


def _training_sampling_config() -> dict[str, int]:
    return {
        "max_rows": max(100_000, int(os.getenv("IDENTIFAI_MAX_TRAIN_MATRIX_ROWS", str(DEFAULT_MAX_TRAIN_ROWS)))),
        "min_mod": max(1, int(os.getenv("IDENTIFAI_MIN_NEGATIVE_SAMPLE_MOD", str(DEFAULT_MIN_NEGATIVE_SAMPLE_MOD)))),
        "max_mod": max(1, int(os.getenv("IDENTIFAI_MAX_NEGATIVE_SAMPLE_MOD", str(DEFAULT_MAX_NEGATIVE_SAMPLE_MOD)))),
    }


def _bounded_training_condition(
    store: BlockingStore,
    where: str,
    parameters: Sequence[object],
) -> tuple[str, tuple[object, ...], dict[str, int]]:
    """Select a deterministic, resource-bounded training subset.

    All available positive labels are retained. Negative/hard-negative rows are
    sampled deterministically from the candidate table when the materialized
    training matrix would otherwise exceed the configured row cap. This bounds
    memory/disk without silently dropping known positives.
    """
    config = _training_sampling_config()
    params = tuple(parameters)
    base = f"({where})" if where else "1=1"
    positive = "truth.target_id IS NOT NULL"
    hard_negative = "(" + " OR ".join((
        "c.rank = 1",
        "c.exact_evidence > 0.0",
        "c.similarity >= 0.82",
        "c.semantic_score >= 0.85",
    )) + ")"

    total = int(store.feature_count(where, params))
    positive_count = int(store.feature_count(f"{base} AND {positive}", params))
    max_rows = int(config["max_rows"])

    if total <= max_rows:
        return (
            where,
            params,
            {**config, "sample_mod": 1, "positive_rows": positive_count, "selected_rows": total},
        )

    if positive_count >= max_rows:
        # This is an extreme dataset/label-density case. Keep every positive row
        # and no additional negatives; correctness is preferable to an arbitrary
        # deletion of labeled matches. The configured cap is a safety target, not a
        # license to throw away truth labels.
        condition = f"{base} AND {positive}"
        selected = positive_count
        LOGGER.warning(
            "All positive labels (%d) exceed the configured training cap (%d); "
            "training on the positive rows only for this slice.",
            positive_count, max_rows,
        )
        return (
            condition,
            params,
            {**config, "sample_mod": 0, "positive_rows": positive_count, "selected_rows": selected},
        )

    negative_budget = max_rows - positive_count
    hard_total = int(store.feature_count(f"{base} AND ({hard_negative}) AND NOT ({positive})", params))
    sample_mod = max(1, int(config["min_mod"]))
    upper = max(sample_mod, int(config["max_mod"]))

    while True:
        # A cheap deterministic row-local hash surrogate. It intentionally uses
        # stable database values rather than Python's randomized hash().
        sampled = (
            "((c.rank * 37 + length(s.entity_id) * 11 + "
            "length(t.entity_id) * 13) % " + str(sample_mod) + ") = 0"
        )
        condition = (
            f"{base} AND (({positive}) OR (({hard_negative}) AND ({sampled})))"
        )
        selected = int(store.feature_count(condition, params))
        if selected <= max_rows or sample_mod >= upper:
            break
        sample_mod = min(upper, sample_mod * 2)

    # If hard negatives are still too abundant at the configured maximum modulus,
    # the selection can still exceed the cap only through correlated deterministic
    # values. Increase the modulus beyond the user ceiling locally until the actual
    # SQL count fits; this is a resource-safety mechanism, not a data-specific rule.
    safety_rounds = 0
    while selected > max_rows and safety_rounds < 16:
        sample_mod *= 2
        safety_rounds += 1
        sampled = (
            "((c.rank * 37 + length(s.entity_id) * 11 + "
            "length(t.entity_id) * 13) % " + str(sample_mod) + ") = 0"
        )
        condition = f"{base} AND (({positive}) OR (({hard_negative}) AND ({sampled})))"
        selected = int(store.feature_count(condition, params))

    LOGGER.info(
        "Bounded training selection: total=%d positives=%d hard_negatives=%d selected=%d cap=%d sample_mod=%d",
        total, positive_count, hard_total, selected, max_rows, sample_mod,
    )
    return (
        condition,
        params,
        {
            **config,
            "sample_mod": sample_mod,
            "positive_rows": positive_count,
            "hard_negative_rows": hard_total,
            "selected_rows": selected,
        },
    )


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
    sampling_meta: dict[str, int] = {}
    if labels and subsample:
        where, parameters, sampling_meta = _bounded_training_condition(store, where, tuple(parameters))

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
                [row[:-1] for row in batch],
                variation_model=variation_model,
                semantic_retriever=store.semantic_retriever,
            )
            if y is not None:
                y[written : written + len(batch)] = [row[-1] for row in batch]
            pair_file.writelines(f"{row[0]}\t{row[1]}\n" for row in batch)
            written += len(batch)

    if hasattr(x, "flush"):
        x.flush()
    if y is not None and hasattr(y, "flush"):
        y.flush()
    if written != rows:
        raise AssertionError(f"Materialization wrote {written} rows but expected {rows}")

    signature_row = store.connection.execute(
        "SELECT value FROM pipeline_meta WHERE key='signature'"
    ).fetchone()
    matrix_meta = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "rows": rows,
        "feature_count": len(FEATURE_NAMES),
        "store_signature": signature_row[0] if signature_row else None,
        "semantic_config": SemanticConfig.from_environment(top_k=store.top_k).signature(),
        "sampling_config": sampling_meta,
        "sampling_environment": _training_sampling_config(),
        "module_signatures": _module_signature(
            "features.py", "variation.py", "semantic_retrieval.py"
        ),
    }
    (scratch / f"{name}.meta.json").write_text(
        json.dumps(matrix_meta, sort_keys=True),
        encoding="utf-8",
    )
    if sampling_meta:
        LOGGER.info(
            "Materialized %s: %d rows x %d features (bounded training sample: %s)",
            name, rows, len(FEATURE_NAMES), sampling_meta,
        )
    else:
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
    """Compute exact macro F0.5 while keeping per-entity state bounded.

    Candidate rows are written in Source-1 order. We therefore stream one
    predicted group at a time and merge it with an ordered Source-1/truth cursor
    instead of materializing several million-ID Python dictionaries.
    """
    labels = matrix.y()
    if labels is None:
        raise ValueError("Grouped F0.5 requires labels")
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if len(probabilities) != matrix.rows or len(labels) != matrix.rows:
        raise ValueError("Probability/label matrix length mismatch")

    truth_where = where.replace("s.", "s_truth.") if where else ""
    truth_predicate = f"WHERE {truth_where}" if truth_where else ""
    source_predicate = f"WHERE {where}" if where else ""
    source_cursor = store.connection.execute(
        f"""
        SELECT s.entity_id,
               COALESCE(tc.truth_count, 0) AS truth_count,
               COALESCE(NULLIF(s.country, ''), '<missing>') AS country
        FROM source1 s
        LEFT JOIN (
            SELECT t.source1_id, COUNT(*) AS truth_count
            FROM truth t
            JOIN source1 s_truth ON s_truth.entity_id=t.source1_id
            {truth_predicate}
            GROUP BY t.source1_id
        ) tc ON tc.source1_id=s.entity_id
        {source_predicate}
        ORDER BY s.entity_id
        """,
        parameters,
    )

    pair_file = matrix.pairs_path.open("r", encoding="utf-8")
    pair_index = 0
    pending_line = pair_file.readline()

    def _next_pair_group():
        nonlocal pending_line, pair_index
        if not pending_line:
            return None
        sid = pending_line.split("\t", 1)[0]
        predicted = 0
        tp = 0
        while pending_line:
            current_sid = pending_line.split("\t", 1)[0]
            if current_sid != sid:
                break
            is_predicted = bool(probabilities[pair_index] >= threshold)
            if is_predicted:
                predicted += 1
                if bool(labels[pair_index]):
                    tp += 1
            pair_index += 1
            pending_line = pair_file.readline()
        return sid, predicted, tp

    score_sum = 0.0
    entity_count = 0
    country_sums: dict[str, float] = {}
    country_counts: dict[str, int] = {}

    try:
        pair_group = _next_pair_group()
        for source_id, truth_count, country in source_cursor:
            while pair_group is not None and pair_group[0] < source_id:
                # A candidate group should always correspond to a Source-1 row;
                # consume defensively rather than allowing desynchronization to
                # abort the entire evaluation.
                pair_group = _next_pair_group()
            if pair_group is not None and pair_group[0] == source_id:
                _, predicted_count, tp = pair_group
                pair_group = _next_pair_group()
            else:
                predicted_count = 0
                tp = 0

            score = float(
                f05_from_counts(
                    tp=int(tp),
                    predicted=int(predicted_count),
                    truth=int(truth_count),
                )
            )
            score_sum += score
            entity_count += 1
            country_sums[country] = country_sums.get(country, 0.0) + score
            country_counts[country] = country_counts.get(country, 0) + 1
    finally:
        source_cursor.close()
        pair_file.close()

    if pair_index != matrix.rows:
        raise AssertionError("Pair/label stream length mismatch")

    return (
        score_sum / entity_count if entity_count else 0.0,
        {
            country: country_sums[country] / country_counts[country]
            for country in country_sums
        },
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


def _write_methodology_documentation() -> Path:
    """Write repository documentation without embedding run-specific measurements."""
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "Documentation.md"
    text = """# Business Entity Resolution Pipeline — Methodology

## Objective

Resolve each Source-1 entity to zero, one, or multiple Source-2/Source-3 entities while preserving the submission contract. The optimization metric is macro F0.5 over Source-1 entities.

## Data handling

The pipeline reads only the supplied challenge TSV files. It does not call external entity databases, geocoders, registries, business APIs, or web enrichment services. Git-LFS pointer files and malformed dataset headers are rejected before processing.

## Preprocessing

Names are Unicode-normalized, case-folded, and canonicalized for common legal-form variants. Addresses are normalized with conservative aliases, structured components, house-number extraction, postal-code extraction, and landmark filler removal. The normalization rules are deterministic and data-independent.

## Candidate generation

Candidate generation combines deterministic lexical blocking with an independent semantic rescue route. Lexical blocking uses exact and near-exact keys, rare-token filtering, phonetic keys, address structure, and MinHash/LSH. Oversized blocks are handled by a relevance-aware rescue stage instead of arbitrary ID truncation.

The semantic route uses a Sentence-Transformers BGE bi-encoder with normalized embeddings and FAISS inner-product search. For large target tables, the index is IVF-PQ with bounded training samples and compressed vector codes. A RAM preflight checks the actual process/container memory budget before semantic resources are loaded. Target IDs use a disk-backed fixed-width memmap for large indexes.

Semantic retrieval is deliberately a rescue path rather than a second full candidate universe: Source-1 entities with enough lexical candidates are not redundantly queried. Semantic query results are streamed directly into SQLite in bounded batches. If optional semantic dependencies, model weights, FAISS, memory, or inference fail, only semantic retrieval is disabled; lexical retrieval and model training continue.

The final candidate set has a configurable hard TOP_K ceiling. Those exact final candidates are the only pairs materialized into the feature matrix and the only pairs eligible for prediction.

## Features

The matcher uses lexical, structured-address, country, retrieval-rank, reciprocal-rank, cross-source corroboration, learned variation, semantic similarity, semantic rank, semantic/lexical alignment, and bounded cross-encoder reranking features.

## Matching model

The model layer supports LightGBM, a linear logistic model, and a probability ensemble. Training matrices are memory-mapped. Training subsets are deterministically bounded for resource safety while retaining all available positive labels and sampling hard negatives.

The tuning stage measures model variants on leakage-safe validation splits and optimizes grouped macro-F0.5 using the same Source-1 grouping used by the metric. Inference uses the tuned decision policy produced by that measured tuning stage.

## Cache correctness

SQLite candidate stores and materialized feature matrices are invalidated when dataset signatures, feature counts, sampling configuration, semantic configuration, or relevant source-module hashes change. Materialized matrices carry sidecar metadata so a stale feature layout cannot be silently reused.

## Reproducibility

All behavior is controlled by code and configuration, not entity-specific IDs or hand-written exceptions. Semantic model names, ANN settings, batch sizes, resource thresholds, and device selection can be overridden through environment variables.

## Runtime requirements

The core pipeline runs without transformer or FAISS packages. The optional semantic path uses `sentence-transformers` and `faiss-cpu` when installed and when compatible local/cached BGE weights are available. Network model downloads are disabled by default so offline training cannot stall on external model fetching. Resource preflight also prevents the semantic index from consuming an unsafe fraction of available/container RAM.

The optional cross-encoder remains disabled by default because transformer reranking is bounded by `IDENTIFAI_RERANK_TOP_K` and should be enabled only after local model availability and runtime have been measured.

## Evaluation artifacts

Real-data validation measurements are written to `scratch/validation_report.json` and tuning results to `scratch/tuned_policy.json`. Repository documentation intentionally does not copy run-specific performance numbers, so synthetic smoke-test results cannot be mistaken for challenge results.
"""
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


def _validate_policy_path(scratch: Path, *, allow_baseline: bool) -> Path:
    report_path = scratch / "validation_report.json"
    baseline_policy_path = scratch / "decision_policy.json"
    tuned_policy_path = scratch / "tuned_policy.json"
    if not report_path.exists() or not baseline_policy_path.exists():
        raise RuntimeError(
            "Prediction is blocked until a completed real validation run exists. "
            "Run `python run.py --mode validate` first."
        )
    if not allow_baseline and not tuned_policy_path.exists():
        raise RuntimeError(
            "Prediction requires the measured tuning stage. "
            "Run `python run.py --mode tune` first, or explicitly pass the "
            "baseline-prediction override for debugging only."
        )
    return tuned_policy_path if tuned_policy_path.exists() else baseline_policy_path


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
        documentation_path = _write_methodology_documentation()
        LOGGER.info("Wrote methodology document: %s", documentation_path)

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
    *,
    allow_baseline: bool = False,
) -> float:
    """Train on all training data and write both required submission files.

    Prediction uses the tuned policy when ``tuned_policy.json`` exists; otherwise
    it falls back to the measured baseline policy.  No prediction is permitted
    before a completed real validation baseline exists.
    """
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    policy_path = _validate_policy_path(scratch, allow_baseline=allow_baseline)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    default_threshold = float(
        policy.get("id_threshold", policy.get("default_threshold", BASELINE_THRESHOLD))
    )
    threshold_by_country = {
        str(country).casefold(): float(value)
        for country, value in (policy.get("thresholds") or policy.get("country_thresholds") or {}).items()
    }
    unseen_threshold = float(
        policy.get(
            "unseen_country_threshold",
            policy.get("zero_shot_fallback_threshold", default_threshold),
        )
    )

    # Decision policies are country-specific when tuning has measured them.
    decision_payload = policy.get("decision_policies") or {}
    default_decision = DecisionPolicy(
        high_threshold=default_threshold,
        low_threshold=default_threshold,
        max_matches=top_k,
        min_absolute_score=0.0,
    )
    decision_policies: dict[str, DecisionPolicy] = {}
    for country, payload in decision_payload.items():
        if isinstance(payload, dict):
            decision_policies[str(country).casefold()] = DecisionPolicy.from_dict(payload)
    if not decision_policies:
        decision_policies = {str(country).casefold(): default_decision for country in threshold_by_country}
    unseen_decision = DecisionPolicy.from_dict(
        policy.get("unseen_decision_policy", default_decision.to_dict())
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
            decision_policies_by_country=decision_policies or None,
            unseen_decision_policy=unseen_decision,
        )
        validate_output_against_store(
            store=test_store,
            candidate_path=candidate_path,
            matching_path=matching_path,
        )
    finally:
        test_store.close()

    return default_threshold
