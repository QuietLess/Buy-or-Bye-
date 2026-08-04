import numpy as np

from src.calibration import SigmoidCalibratedModel


class FakePipeline:
    named_steps = {"preprocessor": object(), "classifier": object()}

    def predict_proba(self, X):
        positive = np.asarray(X, dtype=float).reshape(-1)
        return np.column_stack([1 - positive, positive])


def test_sigmoid_calibrated_model_preserves_probability_shape_and_steps():
    model = SigmoidCalibratedModel(FakePipeline()).fit_calibrator(
        np.array([0.05, 0.2, 0.8, 0.95]), np.array([0, 0, 1, 1])
    )
    probabilities = model.predict_proba(np.array([0.1, 0.9]))
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert model.named_steps is FakePipeline.named_steps
