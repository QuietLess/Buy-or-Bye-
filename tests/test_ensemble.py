import numpy as np

from src.ensemble import CalibratedSoftVotingEnsemble


class FixedModel:
    named_steps = {"classifier": object()}

    def __init__(self, positive):
        self.positive = positive

    def predict_proba(self, X):
        positive = np.full(len(X), self.positive)
        return np.column_stack([1 - positive, positive])


def test_soft_voting_weights_probabilities():
    ensemble = CalibratedSoftVotingEnsemble(FixedModel(.2), FixedModel(.8), .75)
    probabilities = ensemble.predict_proba([1, 2])
    assert np.allclose(probabilities[:, 1], .65)
    assert np.allclose(probabilities.sum(axis=1), 1)
