"""Probability models used by the entity-resolution pipeline.

The model layer is deliberately fail-safe for production data:

* empty/single-class training slices become deterministic constant-probability models;
* LightGBM import/binary/fit failures fall back to sklearn's histogram gradient
  boosting with equivalent sample weighting;
* linear-model fit failures fall back to the observed positive rate;
* all model outputs are finite probabilities in [0, 1].
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
            # A degenerate slice should not crash the whole workflow. Returning 0
            # probability is precision-safe and matches the empty/no-evidence case.
            self.estimator = ("constant", 0.0)
            return True
        unique = np.unique(y)
        if len(unique) == 1:
            self.estimator = ("constant", float(unique[0]))
            return True
        return False

    def _set_prior_fallback(self, y: np.ndarray) -> None:
        prior = float(np.mean(np.asarray(y, dtype=np.float32))) if len(y) else 0.0
        self.estimator = ("constant", float(np.clip(prior, 0.0, 1.0)))

    def _constant_predict(self, X: np.ndarray) -> np.ndarray | None:
        if isinstance(self.estimator, tuple) and self.estimator[0] == "constant":
            return np.full(
                len(X),
                float(np.clip(self.estimator[1], 0.0, 1.0)),
                dtype=np.float32,
            )
        return None


class PairModel(_BaseProbabilityModel):
    """LightGBM classifier with a robust sklearn fallback."""

    def __init__(
        self,
        random_state: int = 42,
        class_weight: str | None = None,
        scale_pos_weight: float | None = None,
    ) -> None:
        super().__init__(random_state)
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight
        self._backend = "unfitted"

    def _fallback_sample_weight(self, y: np.ndarray) -> np.ndarray:
        weights = np.ones(len(y), dtype=np.float64)
        if self.class_weight == "balanced":
            positives = int(np.sum(y == 1))
            negatives = int(np.sum(y == 0))
            if positives and negatives:
                weights[y == 1] = len(y) / (2.0 * positives)
                weights[y == 0] = len(y) / (2.0 * negatives)
        if self.scale_pos_weight is not None:
            weights[y == 1] *= max(0.0, float(self.scale_pos_weight))
        return weights

    def _fit_hist_gradient_boosting(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.estimator = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=self.random_state,
        )
        self.estimator.fit(X, y, sample_weight=self._fallback_sample_weight(y))
        self._backend = "hist_gradient_boosting"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PairModel":
        y = np.asarray(y, dtype=np.int8)
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError(f"PairModel expects a 2-D feature matrix, got {X.shape}")
        if len(X) != len(y):
            raise ValueError("Feature and label row counts differ")
        if self._constant_fit(y):
            self._backend = "constant"
            return self

        try:
            from lightgbm import LGBMClassifier

            estimator = LGBMClassifier(
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
            estimator.fit(X, y)
            self.estimator = estimator
            self._backend = "lightgbm"
            return self
        except Exception as exc:
            LOGGER.warning(
                "LightGBM could not be imported/fitted; using histogram-gradient "
                "fallback (%s: %s)",
                type(exc).__name__,
                exc,
            )

        try:
            self._fit_hist_gradient_boosting(X, y)
            return self
        except Exception as exc:
            LOGGER.warning(
                "Histogram-gradient fallback also failed; using positive-rate "
                "constant model (%s: %s)",
                type(exc).__name__,
                exc,
            )
            self._set_prior_fallback(y)
            self._backend = "constant_fallback"
            return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"PairModel expects a 2-D feature matrix, got {X.shape}")
        constant = self._constant_predict(X)
        if constant is not None:
            return constant

        try:
            values = self.estimator.predict_proba(X)[:, 1].astype(np.float32)
        except Exception as exc:
            LOGGER.warning(
                "PairModel prediction failed; returning conservative zeros (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return np.zeros(len(X), dtype=np.float32)

        return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)

    def describe(self) -> dict[str, object]:
        return {
            "type": self._backend,
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
        self._backend = "unfitted"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearPairModel":
        y = np.asarray(y, dtype=np.int8)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"LinearPairModel expects a 2-D feature matrix, got {X.shape}")
        if len(X) != len(y):
            raise ValueError("Feature and label row counts differ")
        if self._constant_fit(y):
            self._backend = "constant"
            return self

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

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
            self._backend = "logistic_regression"
        except Exception as exc:
            LOGGER.warning(
                "Linear model fit failed; using positive-rate constant fallback "
                "(%s: %s)",
                type(exc).__name__,
                exc,
            )
            self._set_prior_fallback(y)
            self._backend = "constant_fallback"
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"LinearPairModel expects a 2-D feature matrix, got {X.shape}")
        constant = self._constant_predict(X)
        if constant is not None:
            return constant

        try:
            values = self.estimator.predict_proba(X)[:, 1].astype(np.float32)
        except Exception as exc:
            LOGGER.warning(
                "Linear model prediction failed; returning conservative zeros (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return np.zeros(len(X), dtype=np.float32)
        return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)

    def describe(self) -> dict[str, object]:
        return {
            "type": self._backend,
            "class_weight": self.class_weight,
        }


class ProbabilityEnsemble(_BaseProbabilityModel):
    """Weighted probability average of two fitted classifiers."""

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
        self.primary.fit(X, y)
        self.secondary.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        primary = self.primary.predict_proba(X)
        secondary = self.secondary.predict_proba(X)
        return np.clip(
            (
                self.primary_weight * primary
                + (1.0 - self.primary_weight) * secondary
            ).astype(np.float32),
            0.0,
            1.0,
        )

    def describe(self) -> dict[str, object]:
        return {
            "type": "probability_ensemble",
            "primary_weight": self.primary_weight,
            "primary": self.primary.describe(),
            "secondary": self.secondary.describe(),
        }


__all__ = ["LinearPairModel", "PairModel", "ProbabilityEnsemble"]
