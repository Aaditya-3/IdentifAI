"""Model implementations for precision-heavy entity matching.

The production stack uses a LightGBM pair classifier plus an optional structurally
different linear model.  The latter is deliberately small and deterministic so the
ensemble remains cheap enough for the challenge.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


class _BaseProbabilityModel:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = int(random_state)
        self.estimator = None
        self._threshold = 0.5

    @property
    def threshold(self) -> float:
        return float(self._threshold)

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = float(value)

    def predict_pairs(
        self,
        probabilities: Sequence[float],
        pairs: Sequence[tuple[str, str]],
    ) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for (source_id, target_id), probability in zip(pairs, probabilities):
            if float(probability) >= self.threshold:
                result.setdefault(source_id, set()).add(target_id)
        return result

    def _constant_fit(self, y: np.ndarray) -> bool:
        if len(y) == 0:
            self.estimator = None
            return True
        unique = np.unique(y)
        if len(unique) == 1:
            self.estimator = ("constant", float(unique[0]))
            return True
        return False

    def _constant_predict(self, X: np.ndarray) -> np.ndarray | None:
        if self.estimator is None:
            return None
        if isinstance(self.estimator, tuple) and self.estimator[0] == "constant":
            return np.full(len(X), float(self.estimator[1]), dtype=np.float32)
        return None


class PairModel(_BaseProbabilityModel):
    """LightGBM gradient-boosted classifier.

    LightGBM is MIT licensed and comfortably below the challenge's model-size
    constraint.  A small sklearn fallback is retained for environments without
    the optional LightGBM package.
    """

    def __init__(
        self,
        random_state: int = 42,
        class_weight: str | None = None,
        scale_pos_weight: float | None = None,
    ) -> None:
        super().__init__(random_state)
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PairModel":
        y = np.asarray(y, dtype=np.int8)
        if self._constant_fit(y):
            return self

        try:
            from lightgbm import LGBMClassifier

            self.estimator = LGBMClassifier(
                n_estimators=350,
                learning_rate=0.04,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=50,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                class_weight=self.class_weight,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.random_state,
                verbosity=-1,
                n_jobs=-1,
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier

            LOGGER.warning(
                "LightGBM is not installed; using HistGradientBoostingClassifier fallback"
            )
            self.estimator = HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=self.random_state,
            )

        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        constant = self._constant_predict(X)
        if constant is not None:
            return constant
        return self.estimator.predict_proba(X)[:, 1].astype(np.float32)

    def describe(self) -> dict[str, object]:
        return {
            "type": "lightgbm",
            "class_weight": self.class_weight,
            "scale_pos_weight": self.scale_pos_weight,
        }


class LinearPairModel(_BaseProbabilityModel):
    """Standardized logistic regression used as a structurally different model."""

    def __init__(
        self,
        random_state: int = 42,
        class_weight: str | None = None,
    ) -> None:
        super().__init__(random_state)
        self.class_weight = class_weight

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearPairModel":
        y = np.asarray(y, dtype=np.int8)
        if self._constant_fit(y):
            return self

        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Scaling is important because retrieval/rank/count features have a
        # different numeric range from the normalized similarity features.
        self.estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=500,
                class_weight=self.class_weight,
                solver="lbfgs",
                random_state=self.random_state,
            ),
        )
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        constant = self._constant_predict(X)
        if constant is not None:
            return constant
        return self.estimator.predict_proba(X)[:, 1].astype(np.float32)

    def describe(self) -> dict[str, object]:
        return {
            "type": "logistic_regression",
            "class_weight": self.class_weight,
        }


class ProbabilityEnsemble(_BaseProbabilityModel):
    """Weighted probability average of two already-fitted classifiers."""

    def __init__(
        self,
        primary: PairModel,
        secondary: LinearPairModel,
        primary_weight: float = 0.75,
    ) -> None:
        super().__init__(getattr(primary, "random_state", 42))
        weight = float(primary_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("primary_weight must be within [0, 1]")
        self.primary = primary
        self.secondary = secondary
        self.primary_weight = weight

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProbabilityEnsemble":
        # The tuning pipeline constructs the two component models explicitly.
        # fit() is provided only so the class satisfies a normal estimator-like
        # interface; it refits both components deterministically.
        self.primary.fit(X, y)
        self.secondary.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        primary = self.primary.predict_proba(X)
        secondary = self.secondary.predict_proba(X)
        return (
            self.primary_weight * primary
            + (1.0 - self.primary_weight) * secondary
        ).astype(np.float32)

    def describe(self) -> dict[str, object]:
        return {
            "type": "probability_ensemble",
            "primary_weight": self.primary_weight,
            "primary": self.primary.describe(),
            "secondary": self.secondary.describe(),
        }


__all__ = ["LinearPairModel", "PairModel", "ProbabilityEnsemble"]
