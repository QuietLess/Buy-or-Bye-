"""SHAP and model-native explainability helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def _positive_shap_values(values):
    """Normalise SHAP output shapes across supported SHAP versions/models."""
    if isinstance(values, list):
        return np.asarray(values[-1])
    array = np.asarray(values.values if hasattr(values, "values") else values)
    if array.ndim == 3:
        return array[:, :, -1]
    return array


def make_explainer(model, background=None):
    """Create the appropriate SHAP explainer for a fitted pipeline classifier."""
    import shap
    classifier = model.named_steps["classifier"]
    name = classifier.__class__.__name__.lower()
    if "logistic" in name:
        if background is None:
            raise ValueError("LinearExplainer için background verisi gerekir.")
        return shap.LinearExplainer(classifier, background)
    return shap.TreeExplainer(classifier)


def _transform_input(model, input_data):
    """Apply optional feature engineering before fitted preprocessing."""
    steps = model.named_steps
    engineered = steps["feature_engineer"].transform(input_data) if "feature_engineer" in steps else input_data
    return steps["preprocessor"].transform(engineered)


def explain_single_prediction(model, input_data: pd.DataFrame, explainer=None) -> list[tuple[str, float]]:
    """Return the ten transformed features with the largest local SHAP magnitude."""
    preprocessor = model.named_steps["preprocessor"]
    transformed = _transform_input(model, input_data)
    if explainer is None:
        explainer = make_explainer(model, transformed)
    values = _positive_shap_values(explainer(transformed))
    names = preprocessor.get_feature_names_out()
    ranking = sorted(zip(names, values[0]), key=lambda pair: abs(pair[1]), reverse=True)
    return [(name.replace("num__", "").replace("cat__", ""), float(value)) for name, value in ranking[:10]]


def rank_global_shap(values, feature_names, top_n: int = 15) -> pd.DataFrame:
    """Rank transformed features by mean absolute SHAP magnitude."""
    array = _positive_shap_values(values)
    if array.ndim != 2 or array.shape[1] != len(feature_names):
        raise ValueError("SHAP değerleri ile özellik adlarının boyutları uyuşmuyor.")
    table = pd.DataFrame({
        "feature": list(feature_names),
        "mean_abs_shap": np.mean(np.abs(array), axis=0),
        "mean_signed_shap": np.mean(array, axis=0),
    })
    return table.nlargest(top_n, "mean_abs_shap").reset_index(drop=True)


def global_shap_importance(model, X_reference: pd.DataFrame, sample_size: int = 500, top_n: int = 15) -> pd.DataFrame:
    """Return model-wide SHAP importance on a deterministic reference sample."""
    preprocessor = model.named_steps["preprocessor"]
    sample = X_reference.sample(min(sample_size, len(X_reference)), random_state=42)
    transformed = _transform_input(model, sample)
    explainer = make_explainer(model, transformed)
    values = explainer(transformed)
    return rank_global_shap(values, preprocessor.get_feature_names_out(), top_n=top_n)


def plot_shap_summary(model, X_reference: pd.DataFrame, model_name: str = "Model", sample_size: int = 500):
    """Generate the global SHAP beeswarm plot and return its explainer."""
    import shap
    preprocessor = model.named_steps["preprocessor"]
    sample = X_reference.sample(min(sample_size, len(X_reference)), random_state=42)
    transformed = _transform_input(model, sample)
    explainer = make_explainer(model, transformed)
    values = _positive_shap_values(explainer(transformed))
    shap.summary_plot(values, transformed, feature_names=preprocessor.get_feature_names_out(), show=False, max_display=15)
    fig = plt.gcf()
    fig.suptitle(f"SHAP Summary — {model_name}")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return explainer


def plot_feature_importance(model, top_n: int = 20) -> None:
    """Persist model-native global importance as an explainability fallback."""
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return
    names = model.named_steps["preprocessor"].get_feature_names_out()
    table = pd.DataFrame({"feature": names, "importance": classifier.feature_importances_})
    table = table.nlargest(top_n, "importance").sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(table["feature"].str.replace(r"^(num|cat)__", "", regex=True), table["importance"], color="#10b981")
    ax.set(xlabel="Önem", title=f"En Önemli {top_n} Özellik")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
