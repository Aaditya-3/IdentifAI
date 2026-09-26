"""High-precision PairModel classifier adhering to MIT/Apache-2.0 and size constraints."""
from __future__ import annotations

import logging
import numpy as np

LOGGER = logging.getLogger(__name__)

class PairModel:
    def __init__(self, random_state: int = 42, class_weight: str | None = None, scale_pos_weight: float | None = None):
        self.random_state = random_state
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight
        self.estimator = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PairModel":
        if len(y) == 0:
            self.estimator = None
            return self
        unique = np.unique(y)
        if len(unique) == 1:
            self.estimator = ("constant", float(unique[0]))
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
                colsample_bytree=0.85,
                class_weight=self.class_weight,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.random_state,
                verbosity=-1,
                n_jobs=-1,
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            LOGGER.warning("LightGBM not installed; falling back to HistGradientBoostingClassifier")
            self.estimator = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=self.random_state,
            )

        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            return np.zeros(len(X), dtype=np.float32)
        if isinstance(self.estimator, tuple) and self.estimator[0] == "constant":
            return np.full(len(X), self.estimator[1], dtype=np.float32)
        return self.estimator.predict_proba(X)[:, 1].astype(np.float32)