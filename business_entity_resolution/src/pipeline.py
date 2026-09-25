"""Training, local validation, and submission generation."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence, Set, Tuple

import numpy as np

from .blocking import generate_all_candidates
from .data import Labels, load_ground_truth, load_records, split_source1
from .features import feature_matrix
from .metrics import macro_f0_5
from .model import PairModel


def _load_dataset(directory: str | Path, split: str):
    directory = Path(directory)
    sources = [load_records(directory / f"{split}_source{i}.tsv") for i in (1, 2, 3)]
    return sources


def _make_training(source1, source2, source3, truth: Labels, top_k: int = 20):
    candidates = generate_all_candidates(source1, source2, source3, top_k=top_k)
    X, pairs, _ = feature_matrix(candidates, source1, source2, source3)
    y = np.asarray([int(tid in truth.get(sid, set())) for sid, tid in pairs], dtype=np.int8)
    return candidates, X, pairs, y


def validate(data_dir: str | Path, top_k: int = 20, seed: int = 42) -> Tuple[float, float]:
    source1, source2, source3 = _load_dataset(data_dir, "train")
    truth = load_ground_truth(Path(data_dir) / "train_ground_truth.tsv")
    train_s1, valid_s1 = split_source1(source1, truth, seed=seed)
    train_ids = {r["entity_id"] for r in train_s1}
    train_truth = {sid: ids for sid, ids in truth.items() if sid in train_ids}
    _, X_train, _, y_train = _make_training(train_s1, source2, source3, train_truth, top_k)
    valid_truth = {r["entity_id"]: truth.get(r["entity_id"], set()) for r in valid_s1}
    _, X_valid, valid_pairs, _ = _make_training(valid_s1, source2, source3, truth, top_k)
    model = PairModel(seed).fit(X_train, y_train)
    probabilities = model.predict_proba(X_valid)
    threshold = model.tune_threshold(probabilities, valid_pairs, valid_truth)
    predictions = model.predict_pairs(probabilities, valid_pairs)
    return macro_f0_5(valid_truth, predictions), threshold


def train_full(data_dir: str | Path, top_k: int = 20, seed: int = 42):
    source1, source2, source3 = _load_dataset(data_dir, "train")
    truth = load_ground_truth(Path(data_dir) / "train_ground_truth.tsv")
    _, X, _, y = _make_training(source1, source2, source3, truth, top_k)
    return PairModel(seed).fit(X, y)


def predict(test_dir: str | Path, output_dir: str | Path, train_dir: str | Path, top_k: int = 20, seed: int = 42):
    train_source1, train_source2, train_source3 = _load_dataset(train_dir, "train")
    truth = load_ground_truth(Path(train_dir) / "train_ground_truth.tsv")
    _, X_train, _, y_train = _make_training(train_source1, train_source2, train_source3, truth, top_k)
    model = PairModel(seed).fit(X_train, y_train)
    # Tune with a deterministic held-out partition, then refit on all candidates.
    fit_s1, tune_s1 = split_source1(train_source1, truth, seed=seed)
    fit_ids = {r["entity_id"] for r in fit_s1}
    fit_truth = {sid: ids for sid, ids in truth.items() if sid in fit_ids}
    _, X_fit, _, y_fit = _make_training(fit_s1, train_source2, train_source3, fit_truth, top_k)
    _, X_tune, tune_pairs, _ = _make_training(tune_s1, train_source2, train_source3, truth, top_k)
    tuner = PairModel(seed).fit(X_fit, y_fit)
    tune_truth = {r["entity_id"]: truth.get(r["entity_id"], set()) for r in tune_s1}
    model.threshold = tuner.tune_threshold(tuner.predict_proba(X_tune), tune_pairs, tune_truth)

    test_s1, test_s2, test_s3 = _load_dataset(test_dir, "test")
    candidates = generate_all_candidates(test_s1, test_s2, test_s3, top_k=top_k)
    X_test, pairs, _ = feature_matrix(candidates, test_s1, test_s2, test_s3)
    predictions = model.predict_pairs(model.predict_proba(X_test), pairs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(output_dir / "candidate_pairs.tsv", "source1_entity_id", "candidate_entity_ids", _group_candidates(test_s1, candidates))
    _write_tsv(output_dir / "matching_results.tsv", "source1_entity_id", "matched_entity_ids", predictions, preserve_empty=True, order=[r["entity_id"] for r in test_s1])
    candidate_ids = _group_candidates(test_s1, candidates)
    for sid, ids in predictions.items():
        if not ids.issubset(candidate_ids.get(sid, set())):
            raise AssertionError(f"Prediction for {sid} is not a subset of candidate pairs")
    return model.threshold


def _group_candidates(source1, candidates):
    grouped: Dict[str, Set[str]] = {r["entity_id"]: set() for r in source1}
    for sid, tid in candidates:
        grouped.setdefault(sid, set()).add(tid)
    return grouped


def _write_tsv(path: Path, key_name: str, value_name: str, grouped: Dict[str, Set[str]], preserve_empty: bool = True, order=None):
    order = list(order) if order is not None else list(grouped)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow([key_name, value_name])
        for sid in order:
            ids = sorted(grouped.get(sid, set()))
            writer.writerow([sid, ",".join(ids) if ids or preserve_empty else ""])

