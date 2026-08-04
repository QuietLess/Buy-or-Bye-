import numpy as np
import pytest

from src.explain import rank_global_shap


def test_global_shap_ranking_uses_mean_absolute_magnitude():
    values = np.array([[1.0, -4.0, .1], [-1.0, 2.0, -.1]])
    result = rank_global_shap(values, ["a", "b", "c"], top_n=2)
    assert result["feature"].tolist() == ["b", "a"]
    assert result.iloc[0]["mean_abs_shap"] == pytest.approx(3.0)


def test_global_shap_ranking_validates_shape():
    with pytest.raises(ValueError, match="boyutları"):
        rank_global_shap(np.ones((2, 3)), ["a", "b"])
