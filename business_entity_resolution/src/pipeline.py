"""Training, country-holdout validation, and submission generation."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .blocking import generate_all_candidates
from .data import Labels, load_ground_truth, load_records, split_country_holdout
from .features import feature_matrix
from .metrics import macro_f0_5
from .model import PairModel
from .output import write_error_csv, write_submission


def _load_dataset(directory: str | Path, split: str):
    directory = Path(directory)
    return [load_records(directory / f"{split}_source{i}.tsv") for i in (1, 2, 3)]


def _truth_for(records, truth: Labels) -> Labels:
    return {record["entity_id"]: set(truth.get(record["entity_id"], set())) for record in records}


def _make_training(source1, source2, source3, truth: Labels, top_k: int = 20):
    candidates = generate_all_candidates(source1, source2, source3, top_k=top_k, truth=truth)
    X, pairs, _ = feature_matrix(candidates, source1, source2, source3)
    y = np.asarray([int(tid in truth.get(sid, set())) for sid, tid in pairs], dtype=np.int8)
    return candidates, X, pairs, y


def _country_holdout(source1, truth: Labels, train_country=None, validation_country=None):
    train_s1, valid_s1 = split_country_holdout(
        source1,
        train_country=train_country,
        validation_country=validation_country,
    )
    return train_s1, valid_s1, _truth_for(train_s1, truth), _truth_for(valid_s1, truth)


def validate(
    data_dir: str | Path,
    top_k: int = 20,
    seed: int = 42,
    train_country: str | None = None,
    validation_country: str | None = None,
    error_path: str | Path | None = None,
) -> Tuple[float, float]:
    """Fit on one country and tune/evaluate on a held-out country."""
    source1, source2, source3 = _load_dataset(data_dir, "train")
    truth = load_ground_truth(Path(data_dir) / "train_ground_truth.tsv")
    train_s1, valid_s1, train_truth, valid_truth = _country_holdout(
        source1, truth, train_country, validation_country
    )
    _, X_train, _, y_train = _make_training(train_s1, source2, source3, train_truth, top_k)
    candidates, X_valid, valid_pairs, _ = _make_training(valid_s1, source2, source3, valid_truth, top_k)
    model = PairModel(seed).fit(X_train, y_train)
    probabilities = model.predict_proba(X_valid)
    threshold = model.tune_threshold(probabilities, valid_pairs, valid_truth)
    predictions = model.predict_pairs(probabilities, valid_pairs)
    score = macro_f0_5(valid_truth, predictions, (record["entity_id"] for record in valid_s1))

    error_path = Path(error_path) if error_path is not None else Path(data_dir) / "validation_errors.csv"
    write_error_csv(error_path, valid_truth, predictions, candidates, valid_pairs, probabilities, threshold)
    return score, threshold


def train_full(data_dir: str | Path, top_k: int = 20, seed: int = 42):
    source1, source2, source3 = _load_dataset(data_dir, "train")
    truth = load_ground_truth(Path(data_dir) / "train_ground_truth.tsv")
    _, X, _, y = _make_training(source1, source2, source3, _truth_for(source1, truth), top_k)
    return PairModel(seed).fit(X, y)


def predict(
    test_dir: str | Path,
    output_dir: str | Path,
    train_dir: str | Path,
    top_k: int = 20,
    seed: int = 42,
    train_country: str | None = None,
    validation_country: str | None = None,
):
    train_source1, train_source2, train_source3 = _load_dataset(train_dir, "train")
    truth = load_ground_truth(Path(train_dir) / "train_ground_truth.tsv")
    truth = _truth_for(train_source1, truth)

    _, X_train, _, y_train = _make_training(train_source1, train_source2, train_source3, truth, top_k)
    model = PairModel(seed).fit(X_train, y_train)

    fit_s1, tune_s1, fit_truth, tune_truth = _country_holdout(
        train_source1, truth, train_country, validation_country
    )
    _, X_fit, _, y_fit = _make_training(fit_s1, train_source2, train_source3, fit_truth, top_k)
    _, X_tune, tune_pairs, _ = _make_training(tune_s1, train_source2, train_source3, tune_truth, top_k)
    tuner = PairModel(seed).fit(X_fit, y_fit)
    model.threshold = tuner.tune_threshold(tuner.predict_proba(X_tune), tune_pairs, tune_truth)

    test_s1, test_s2, test_s3 = _load_dataset(test_dir, "test")
    candidates = generate_all_candidates(test_s1, test_s2, test_s3, top_k=top_k)
    X_test, pairs, _ = feature_matrix(candidates, test_s1, test_s2, test_s3)
    predictions = model.predict_pairs(model.predict_proba(X_test), pairs)
    write_submission(output_dir, test_s1, candidates, predictions)
    return model.threshold
