"""Testable helpers used by the Streamlit inference dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from .data import TARGET
from .predict import predict_sessions, prepare_input


def normalise_target(series: pd.Series) -> pd.Series:
    """Normalise common CSV representations of the Revenue label."""
    mapping = {
        True: 1, False: 0, 1: 1, 0: 0,
        "True": 1, "False": 0, "TRUE": 1, "FALSE": 0,
        "true": 1, "false": 0, "1": 1, "0": 0,
    }
    target = series.map(mapping)
    if target.isna().any():
        invalid = sorted(map(str, series[target.isna()].dropna().unique()))
        raise ValueError(f"Revenue yalnızca 0/1 veya True/False olabilir: {invalid}")
    return target.astype(int)


def score_batch(data: pd.DataFrame, model, metadata: dict) -> tuple[pd.DataFrame, dict | None]:
    """Score uploaded rows and calculate classification metrics when labels exist."""
    if data.empty:
        raise ValueError("Yüklenen CSV boş.")
    features = prepare_input(data)
    scored = predict_sessions(features, model, metadata)
    output = data.copy()
    for column in scored.columns:
        output[column] = scored[column].to_numpy()

    if TARGET not in data.columns:
        return output, None

    y_true = normalise_target(data[TARGET])
    probability = scored["purchase_probability"].to_numpy()
    prediction = scored["purchase_prediction"].to_numpy()
    metrics = {
        "rows": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
        "roc_auc": None,
        "pr_auc": None,
    }
    if y_true.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probability))
        metrics["pr_auc"] = float(average_precision_score(y_true, probability))
    return output, metrics


def calculate_what_if(
    data: pd.DataFrame, model, feature: str, lower: float, upper: float, steps: int = 40,
) -> pd.DataFrame:
    """Vary one numeric feature while holding all other session values constant."""
    if feature not in data.columns:
        raise ValueError(f"What-if özelliği girdide bulunamadı: {feature}")
    if lower >= upper:
        raise ValueError("What-if alt sınırı üst sınırdan küçük olmalı.")
    values = np.linspace(float(lower), float(upper), int(steps))
    scenarios = pd.concat([data.iloc[[0]]] * len(values), ignore_index=True)
    scenarios[feature] = values
    probabilities = np.asarray(model.predict_proba(prepare_input(scenarios)))[:, 1]
    return pd.DataFrame({feature: values, "purchase_probability": probabilities})
