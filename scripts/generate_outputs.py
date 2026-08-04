"""Regenerate every static chart from existing data and model artifacts."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from src.data import load_data
from src.evaluate import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_threshold_optimization,
)
from src.explain import plot_feature_importance, plot_shap_summary
from src.predict import load_artifacts
from src.train import split_data


def main() -> None:
    data = load_data()
    X_train, X_val, X_test, _, y_val, y_test = split_data(data)
    model, metadata = load_artifacts()
    model_name = metadata["model_name"]
    explanation_name = (
        f"Engineered LightGBM component ({model.lightgbm_weight:.0%} ensemble weight)"
        if hasattr(model, "lightgbm_weight") else model_name
    )
    threshold = float(metadata["threshold"])
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    plot_confusion_matrix(y_test, test_proba, threshold, model_name)
    plot_pr_curve(y_test, test_proba, model_name)
    plot_roc_curve(y_test, test_proba, model_name)
    plot_threshold_optimization(y_val, val_proba, model_name)
    plot_feature_importance(model)
    plot_shap_summary(model, X_train, explanation_name)

    from src.segment_analysis import main as generate_segment_report
    generate_segment_report()

    from scripts.generate_deliverables import generate_eda, generate_notebook, generate_slides
    generate_eda()
    generate_notebook()
    generate_slides()
    print("Tüm grafikler outputs/ klasöründe oluşturuldu.")


if __name__ == "__main__":
    main()
