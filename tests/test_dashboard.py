import numpy as np
import pandas as pd
import pytest

from src.dashboard import calculate_what_if, normalise_target, score_batch
from src.data import FEATURE_COLUMNS


class LinearDummyModel:
    def predict_proba(self, frame):
        probability = np.clip(np.asarray(frame["PageValues"], dtype=float) / 100, 0, 1)
        return np.column_stack([1 - probability, probability])


def sample_frame(rows=2):
    frame = pd.DataFrame([{column: 0 for column in FEATURE_COLUMNS} for _ in range(rows)])
    frame["PageValues"] = [10, 80][:rows]
    return frame


def test_score_batch_with_labels_returns_predictions_and_metrics():
    frame = sample_frame()
    frame["Revenue"] = [False, True]
    output, metrics = score_batch(frame, LinearDummyModel(), {"threshold": 0.26})
    assert list(output["segment"]) == ["LOW", "HIGH"]
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_score_batch_without_labels_has_no_metrics():
    output, metrics = score_batch(sample_frame(), LinearDummyModel(), {"threshold": 0.26})
    assert metrics is None
    assert "purchase_probability" in output


def test_what_if_changes_only_selected_feature_grid():
    result = calculate_what_if(sample_frame(1), LinearDummyModel(), "PageValues", 0, 100, 5)
    assert result["purchase_probability"].tolist() == pytest.approx([0, .25, .5, .75, 1])


def test_target_normalisation_rejects_unknown_values():
    with pytest.raises(ValueError, match="Revenue"):
        normalise_target(pd.Series(["yes"]))
