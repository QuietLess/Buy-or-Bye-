"""Persist global SHAP importance for fast use by the Streamlit dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .data import load_data
from .explain import global_shap_importance
from .predict import load_artifacts
from .train import split_data
from .ui_labels import feature_label

ROOT = Path(__file__).resolve().parents[1]


def main() -> dict:
    data = load_data()
    X_train, _, _, _, _, _ = split_data(data)
    model, metadata = load_artifacts()
    table = global_shap_importance(model, X_train, sample_size=500, top_n=15)
    rows = []
    for row in table.to_dict("records"):
        rows.append({
            "feature": row["feature"],
            "display_name": feature_label(row["feature"]),
            "mean_abs_shap": float(row["mean_abs_shap"]),
            "mean_signed_shap": float(row["mean_signed_shap"]),
        })
    is_ensemble = hasattr(model, "engineered_lightgbm") and hasattr(model, "lightgbm_weight")
    report = {
        "model_name": metadata["model_name"],
        "trained_at": metadata.get("trained_at"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference": "Deterministic 500-row sample from training split",
        "explanation_scope": "engineered_lightgbm_component" if is_ensemble else "full_model",
        "explained_component_weight": float(model.lightgbm_weight) if is_ensemble else 1.0,
        "interpretation": (
            "For the ensemble, SHAP explains only the engineered LightGBM component; "
            "mean_abs_shap ranks component reliance and is not probability or causality."
            if is_ensemble else
            "mean_abs_shap ranks global reliance; signed SHAP is raw model-score contribution, not probability or causality."
        ),
        "features": rows,
    }
    path = ROOT / "reports" / "global_shap_importance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
