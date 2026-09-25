"""Pair classifier and singleton-aware F0.5 threshold selection."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Set, Tuple

import numpy as np

from .metrics import macro_f0_5


class PairModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.estimator = None
        self.threshold = 0.5
        self.best_params: dict[str, float | int] | None = None
        self.best_iteration: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: tuple[np.ndarray, np.ndarray] | None = None) -> "PairModel":
        if len(X) != len(y):
            raise ValueError("Feature rows and labels must have the same length")
        if len(y) == 0:
            self.estimator = None
            return self
        unique = np.unique(y)
        if len(unique) == 1:
            self.estimator = ("constant", float(unique[0]))
            return self
        try:
            from lightgbm import LGBMClassifier, early_stopping
            from sklearn.metrics import average_precision_score
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            raise RuntimeError("LightGBM is required for pair classification") from exc
        if eval_set is None:
            class_counts = np.bincount(y.astype(np.int64), minlength=2)
            # Very small diagnostic datasets cannot be stratified; retain the
            # old direct-fit behavior in that edge case.
            if len(y) >= 20 and np.min(class_counts[class_counts > 0]) >= 2:
                indices = np.arange(len(y))
                train_idx, eval_idx = train_test_split(
                    indices, test_size=0.15, stratify=y, random_state=self.random_state,
                )
                X_train, y_train, X_eval, y_eval = X[train_idx], y[train_idx], X[eval_idx], y[eval_idx]
            else:
                X_train, y_train, X_eval, y_eval = X, y, X, y
        else:
            X_train, y_train = X, y
            X_eval, y_eval = eval_set
            if len(X_eval) != len(y_eval) or len(y_eval) == 0:
                raise ValueError("eval_set must contain equally sized, non-empty feature and label arrays")

        # This compact sweep tunes tree capacity and regularization.  Each model
        # selects its number of trees using a held-out eval_set, then the chosen
        # configuration is refit on all supplied training rows below.
        parameter_grid = (
            {"num_leaves": 15, "min_child_samples": 20, "feature_fraction": 0.90},
            {"num_leaves": 31, "min_child_samples": 30, "feature_fraction": 0.85},
            {"num_leaves": 63, "min_child_samples": 60, "feature_fraction": 0.80},
        )
        best_score, best_params, best_iteration = -np.inf, parameter_grid[0], 100
        for params in parameter_grid:
            candidate = LGBMClassifier(
                objective="binary", n_estimators=2_000, learning_rate=0.04,
                class_weight="balanced", random_state=self.random_state, verbosity=-1,
                n_jobs=-1, **params,
            )
            candidate.fit(
                X_train, y_train, eval_set=[(X_eval, y_eval)], eval_metric="binary_logloss",
                callbacks=[early_stopping(stopping_rounds=75, verbose=False)],
            )
            score = average_precision_score(y_eval, candidate.predict_proba(X_eval)[:, 1])
            if score > best_score:
                best_score, best_params = score, params
                best_iteration = int(candidate.best_iteration_ or candidate.n_estimators_)

        self.best_params = dict(best_params)
        self.best_iteration = best_iteration
        self.estimator = LGBMClassifier(
            objective="binary", n_estimators=best_iteration, learning_rate=0.04,
            class_weight="balanced", random_state=self.random_state, verbosity=-1,
            n_jobs=-1, **best_params,
        )
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            return np.zeros(len(X), dtype=float)
        if isinstance(self.estimator, tuple):
            return np.full(len(X), self.estimator[1], dtype=float)
        return self.estimator.predict_proba(X)[:, 1]

    def tune_threshold(self, probabilities: Sequence[float], pairs: Sequence[Tuple[str, str]], y_true: Mapping[str, Set[str]],
                       thresholds: Sequence[float] | None = None) -> float:
        if len(probabilities) != len(pairs):
            raise ValueError("Probabilities and candidate pairs must have the same length")
        thresholds = thresholds if thresholds is not None else np.linspace(0.30, 0.90, 61)
        if len(thresholds) == 0:
            raise ValueError("At least one threshold is required")
        best_score, best_threshold = -1.0, 0.90
        for threshold in thresholds:
            predictions: Dict[str, Set[str]] = {sid: set() for sid in y_true}
            for (sid, tid), probability in zip(pairs, probabilities):
                if probability >= threshold:
                    predictions.setdefault(sid, set()).add(tid)
            score = macro_f0_5(y_true, predictions)
            if score > best_score or (score == best_score and threshold > best_threshold):
                best_score, best_threshold = score, float(threshold)
        self.threshold = best_threshold
        return best_threshold

    def predict_pairs(self, probabilities: Sequence[float], pairs: Sequence[Tuple[str, str]]) -> Dict[str, Set[str]]:
        if len(probabilities) != len(pairs):
            raise ValueError("Probabilities and candidate pairs must have the same length")
        result: Dict[str, Set[str]] = {}
        for pair, probability in zip(pairs, probabilities):
            if probability >= self.threshold:
                result.setdefault(pair[0], set()).add(pair[1])
        return result
