import numpy as np

from src.evaluate import evaluate_model, find_optimal_threshold, threshold_scores


def test_threshold_search_range_and_metric_output():
    y = np.array([0, 0, 1, 1])
    p = np.array([.05, .2, .7, .9])
    threshold = find_optimal_threshold(y, p)
    assert .05 <= threshold <= .94
    assert len(threshold_scores(y, p)) == 90
    metrics = evaluate_model(y, p, "test", threshold)
    assert metrics["f1_optimized"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
