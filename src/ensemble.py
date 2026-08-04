"""Soft-voting ensemble models used by the final inference layer."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class CalibratedSoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """Blend calibrated RF and engineered LightGBM probabilities."""

    def __init__(self, random_forest, engineered_lightgbm, lightgbm_weight: float):
        self.random_forest = random_forest
        self.engineered_lightgbm = engineered_lightgbm
        self.lightgbm_weight = float(lightgbm_weight)
        self.classes_ = np.array([0, 1])

    @property
    def named_steps(self):
        """Delegate explanations to the engineered LightGBM component."""
        return self.engineered_lightgbm.named_steps

    def predict_proba(self, X):
        rf_positive = self.random_forest.predict_proba(X)[:, 1]
        lgb_positive = self.engineered_lightgbm.predict_proba(X)[:, 1]
        positive = (1 - self.lightgbm_weight) * rf_positive + self.lightgbm_weight * lgb_positive
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
