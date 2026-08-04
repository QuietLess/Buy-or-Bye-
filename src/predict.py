"""Artifact loading and inference helpers shared by the UI and integrations."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS
from .segments import segment_probabilities, segment_probability

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
DEFAULT_THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"


def load_artifacts(model_path=DEFAULT_MODEL_PATH, threshold_path=DEFAULT_THRESHOLD_PATH):
    """Load the trained pipeline and its versioned decision metadata."""
    model_path, threshold_path = Path(model_path), Path(threshold_path)
    if not model_path.exists() or not threshold_path.exists():
        raise FileNotFoundError("Model artifact'leri bulunamadı. Önce `python -m src.train` çalıştırın.")
    model = joblib.load(model_path)
    with threshold_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return model, metadata


def prepare_input(data: pd.DataFrame | dict) -> pd.DataFrame:
    """Put an input record into the exact training schema."""
    frame = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Tahmin girdisinde eksik alanlar var: {missing}")
    return frame.loc[:, FEATURE_COLUMNS]


def predict_session(data, model=None, metadata=None) -> dict:
    """Predict calibrated purchase probability, segment, and business action."""
    if model is None or metadata is None:
        model, metadata = load_artifacts()
    frame = prepare_input(data)
    probability = float(np.asarray(model.predict_proba(frame))[0, 1])
    threshold = float(metadata["threshold"])
    segment = segment_probability(probability, threshold)
    return {
        "probability": probability,
        "threshold": threshold,
        "purchase_prediction": int(probability >= threshold),
        "segment": segment["code"],
        "segment_label": segment["label"],
        "segment_range": [segment["lower"], min(segment["upper"], 1.0)],
        "business_action": segment["action"],
        "business_objective": segment["objective"],
        "segment_color": segment["color"],
    }


def predict_sessions(data, model=None, metadata=None) -> pd.DataFrame:
    """Score a batch once and return probability, segment, and action columns."""
    if model is None or metadata is None:
        model, metadata = load_artifacts()
    frame = prepare_input(data)
    probabilities = np.asarray(model.predict_proba(frame))[:, 1]
    threshold = float(metadata["threshold"])
    segments = segment_probabilities(probabilities, threshold)
    return pd.DataFrame({
        "purchase_probability": probabilities,
        "purchase_prediction": (probabilities >= threshold).astype(int),
        "segment": [row["code"] for row in segments],
        "segment_label": [row["label"] for row in segments],
        "business_action": [row["action"] for row in segments],
        "business_objective": [row["objective"] for row in segments],
    }, index=frame.index)
