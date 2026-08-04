import numpy as np

from src.tune import choose_thresholds


def test_cost_threshold_penalises_false_negatives():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.45, 0.9])
    result = choose_thresholds(y, probabilities, false_positive_cost=1, false_negative_cost=3)
    assert result["cost_optimal"]["fn"] == 0
    assert "f1_optimal" in result
