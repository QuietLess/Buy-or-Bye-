"""Deterministic, leakage-safe session-level feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .preprocess import build_preprocessor

ENGINEERED_NUMERIC_FEATURES = [
    "TotalPages", "TotalDuration", "AdminDurationPerPage",
    "InfoDurationPerPage", "ProductDurationPerPage", "DurationPerPage",
    "ProductPageShare", "AdminPageShare", "InfoPageShare",
    "ExitBounceGap", "RetentionScore", "EngagementScore",
    "ProductEngagement", "LogAdministrativeDuration",
    "LogInformationalDuration", "LogProductDuration", "LogTotalDuration",
    "LogProductPages", "LogPageValues", "MonthSin", "MonthCos",
    "HasAdministrative", "HasInformational", "HasPageValue",
    "WeekendSpecialDay", "ReturningWeekend", "PageValuePerProductPage",
]

ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES


def _safe_divide(numerator, denominator):
    return np.divide(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float),
        out=np.zeros(len(numerator), dtype=float), where=np.asarray(denominator) != 0,
    )


class SessionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Append domain features without learning from labels or global data."""

    MONTH_ORDER = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5,
        "June": 6, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9,
        "Oct": 10, "Nov": 11, "Dec": 12,
    }

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        frame = X.copy()
        index = frame.index

        def column(name, default=0.0):
            return frame[name] if name in frame else pd.Series(default, index=index)

        administrative = pd.to_numeric(column("Administrative"), errors="coerce").fillna(0)
        informational = pd.to_numeric(column("Informational"), errors="coerce").fillna(0)
        product = pd.to_numeric(column("ProductRelated"), errors="coerce").fillna(0)
        admin_duration = pd.to_numeric(column("Administrative_Duration"), errors="coerce").fillna(0)
        info_duration = pd.to_numeric(column("Informational_Duration"), errors="coerce").fillna(0)
        product_duration = pd.to_numeric(column("ProductRelated_Duration"), errors="coerce").fillna(0)
        bounce = pd.to_numeric(column("BounceRates"), errors="coerce").fillna(0)
        exit_rate = pd.to_numeric(column("ExitRates"), errors="coerce").fillna(0)
        page_values = pd.to_numeric(column("PageValues"), errors="coerce").fillna(0)
        special_day = pd.to_numeric(column("SpecialDay"), errors="coerce").fillna(0)

        total_pages = administrative + informational + product
        total_duration = admin_duration + info_duration + product_duration
        frame["TotalPages"] = total_pages
        frame["TotalDuration"] = total_duration
        frame["AdminDurationPerPage"] = _safe_divide(admin_duration, administrative)
        frame["InfoDurationPerPage"] = _safe_divide(info_duration, informational)
        frame["ProductDurationPerPage"] = _safe_divide(product_duration, product)
        frame["DurationPerPage"] = _safe_divide(total_duration, total_pages)
        frame["ProductPageShare"] = _safe_divide(product, total_pages)
        frame["AdminPageShare"] = _safe_divide(administrative, total_pages)
        frame["InfoPageShare"] = _safe_divide(informational, total_pages)
        frame["ExitBounceGap"] = exit_rate - bounce
        frame["RetentionScore"] = 1 - ((bounce + exit_rate) / 2)
        frame["EngagementScore"] = np.log1p(total_duration.clip(lower=0)) * (1 - bounce) * (1 - exit_rate)
        frame["ProductEngagement"] = np.log1p(product_duration.clip(lower=0)) * frame["ProductPageShare"]
        frame["LogAdministrativeDuration"] = np.log1p(admin_duration.clip(lower=0))
        frame["LogInformationalDuration"] = np.log1p(info_duration.clip(lower=0))
        frame["LogProductDuration"] = np.log1p(product_duration.clip(lower=0))
        frame["LogTotalDuration"] = np.log1p(total_duration.clip(lower=0))
        frame["LogProductPages"] = np.log1p(product.clip(lower=0))
        frame["LogPageValues"] = np.log1p(page_values.clip(lower=0))
        month_number = column("Month", "Jan").map(self.MONTH_ORDER).fillna(1).astype(float)
        frame["MonthSin"] = np.sin(2 * np.pi * month_number / 12)
        frame["MonthCos"] = np.cos(2 * np.pi * month_number / 12)
        frame["HasAdministrative"] = (administrative > 0).astype(int)
        frame["HasInformational"] = (informational > 0).astype(int)
        frame["HasPageValue"] = (page_values > 0).astype(int)
        weekend = column("Weekend", False).astype(bool).astype(int)
        returning = column("VisitorType", "Other").eq("Returning_Visitor").astype(int)
        frame["WeekendSpecialDay"] = weekend * special_day
        frame["ReturningWeekend"] = returning * weekend
        frame["PageValuePerProductPage"] = _safe_divide(page_values, product)

        frame[ENGINEERED_NUMERIC_FEATURES] = frame[ENGINEERED_NUMERIC_FEATURES].replace(
            [np.inf, -np.inf], 0,
        ).fillna(0)
        return frame

    def get_feature_names_out(self, input_features=None):
        base = list(input_features if input_features is not None else self.feature_names_in_)
        return np.asarray(base + ENGINEERED_NUMERIC_FEATURES, dtype=object)


def build_engineered_pipeline(base_pipeline) -> Pipeline:
    """Wrap a candidate classifier with feature engineering and preprocessing."""
    return Pipeline([
        ("feature_engineer", SessionFeatureEngineer()),
        ("preprocessor", build_preprocessor(ALL_NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
        ("classifier", clone(base_pipeline.named_steps["classifier"])),
    ])
