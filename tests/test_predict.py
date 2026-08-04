import pandas as pd
import pytest

from src.data import FEATURE_COLUMNS
from src.predict import predict_session, prepare_input


class FixedProbabilityModel:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, frame):
        return [[1 - self.probability, self.probability]]


def test_prepare_input_orders_columns():
    data = {name: 0 for name in reversed(FEATURE_COLUMNS)}
    result = prepare_input(data)
    assert list(result.columns) == FEATURE_COLUMNS


def test_prepare_input_rejects_incomplete_input():
    with pytest.raises(ValueError, match="eksik alan"):
        prepare_input(pd.DataFrame([{"Month": "Nov"}]))


def test_predict_session_returns_probability_segment_and_action():
    data = {name: 0 for name in FEATURE_COLUMNS}
    result = predict_session(data, FixedProbabilityModel(0.30), {"threshold": 0.26})
    assert result["probability"] == pytest.approx(0.30)
    assert result["purchase_prediction"] == 1
    assert result["segment"] == "HIGH"
    assert result["business_action"]
