import numpy as np
import pandas as pd

from src.features import ENGINEERED_NUMERIC_FEATURES, SessionFeatureEngineer


def test_session_feature_engineering_is_finite_and_correct():
    frame = pd.DataFrame([{
        "Administrative": 2, "Administrative_Duration": 10,
        "Informational": 0, "Informational_Duration": 0,
        "ProductRelated": 3, "ProductRelated_Duration": 30,
        "BounceRates": .1, "ExitRates": .2, "PageValues": 12,
        "SpecialDay": .5, "Month": "Dec", "Weekend": True,
        "VisitorType": "Returning_Visitor",
    }])
    result = SessionFeatureEngineer().fit_transform(frame)
    assert result.loc[0, "TotalPages"] == 5
    assert result.loc[0, "TotalDuration"] == 40
    assert result.loc[0, "ProductDurationPerPage"] == 10
    assert result.loc[0, "InfoDurationPerPage"] == 0
    assert result.loc[0, "ReturningWeekend"] == 1
    assert np.isfinite(result[ENGINEERED_NUMERIC_FEATURES].to_numpy()).all()
