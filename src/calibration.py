"""Probability calibration wrappers that preserve pipeline explainability."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


def _logit(probabilities) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(values / (1 - values)).reshape(-1, 1)


class SigmoidCalibratedModel(BaseEstimator, ClassifierMixin):
    """Apply OOF Platt scaling to a fitted classifier pipeline.

    ``named_steps`` delegates to the underlying pipeline so SHAP and the
    Streamlit explanation layer remain compatible with the calibrated model.
    """

    def __init__(self, base_model, calibrator=None):
        self.base_model = base_model
        self.calibrator = calibrator or LogisticRegression(random_state=42)
        self.classes_ = np.array([0, 1])

    @property
    def named_steps(self):
        return self.base_model.named_steps

    def fit_calibrator(self, probabilities, y_true):
        self.calibrator.fit(_logit(probabilities), np.asarray(y_true))
        return self

    def predict_proba(self, X):
        raw_positive = self.base_model.predict_proba(X)[:, 1]
        positive = self.calibrator.predict_proba(_logit(raw_positive))[:, 1]
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
