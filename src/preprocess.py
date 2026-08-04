"""Leakage-safe preprocessing pipeline."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_cols: list[str], categorical_cols: list[str]
) -> ColumnTransformer:
    """Build numeric and categorical transformations fitted only on training data."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, numeric_cols), ("cat", categorical, categorical_cols)],
        verbose_feature_names_out=True,
    )
