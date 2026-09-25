"""Pair classifier and singleton-aware F0.5 threshold selection."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Set, Tuple

import numpy as np

from .metrics import macro_f0_5


class PairModel:
    def __init__(self, random_state: int = 42, class_weight: str | None = None,
                 scale_pos_weight: float | None = None):
        self.random_state = random_state
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight
        self.estimator = None
        self.threshold = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PairModel":
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
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM is required for pair classification") from exc
        options = dict(
            objective="binary",
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=50,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1,
        )
        if self.class_weight is not None:
            options["class_weight"] = self.class_weight
        if self.scale_pos_weight is not None:
            options["scale_pos_weight"] = self.scale_pos_weight
        self.estimator = LGBMClassifier(**options)
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
        thresholds = thresholds if thresholds is not None else np.unique(np.concatenate((
            np.arange(0.30, 0.901, 0.01), np.arange(0.901, 0.996, 0.001),
        )))
        if len(thresholds) == 0:
            raise ValueError("At least one threshold is required")
        best_score, best_threshold = -1.0, 0.995
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
