"""Evaluate a calibrated RF + engineered LightGBM soft-voting ensemble."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .calibration import SigmoidCalibratedModel
from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_data
from .ensemble import CalibratedSoftVotingEnsemble
from .evaluate import evaluate_model
from .features import build_engineered_pipeline
from .train import MODELS_DIR, REPORTS_DIR, build_lightgbm, build_random_forest, split_data
from .tune import calibration_metrics, choose_thresholds

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"


def _candidate_models():
    tuning = json.loads((REPORTS_DIR / "tuning_report.json").read_text(encoding="utf-8"))
    feature = json.loads((REPORTS_DIR / "feature_engineering_report.json").read_text(encoding="utf-8"))
    tuned = {row["model"]: row for row in tuning["cv_results"]}
    engineered = {row["model"]: row for row in feature["tuned_engineered_results"]}

    forest = build_random_forest(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    forest.set_params(**tuned["Random Forest"]["best_params"])
    lightgbm = build_engineered_pipeline(build_lightgbm(NUMERIC_FEATURES, CATEGORICAL_FEATURES))
    lightgbm.set_params(**engineered["Engineered LightGBM"]["best_params"])
    if feature.get("best_iteration_early_stopping"):
        lightgbm.set_params(classifier__n_estimators=int(feature["best_iteration_early_stopping"]))
    return forest, lightgbm, feature


def _fit_calibrated(model, X_train, y_train, cv):
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    fitted = clone(model).fit(X_train, y_train)
    calibrated = SigmoidCalibratedModel(fitted).fit_calibrator(oof, y_train)
    return calibrated, fitted, oof


def _choose_weight(y_true, rf_proba, lgb_proba):
    rows = []
    for weight in np.arange(0, 1.01, .05):
        blended = (1 - weight) * rf_proba + weight * lgb_proba
        rows.append({
            "lightgbm_weight": round(float(weight), 2),
            "random_forest_weight": round(float(1 - weight), 2),
            "pr_auc": float(average_precision_score(y_true, blended)),
            "roc_auc": float(roc_auc_score(y_true, blended)),
        })
    return rows, max(rows, key=lambda row: row["pr_auc"])


def _plot_weight_search(rows):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    best = max(rows, key=lambda row: row["pr_auc"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([row["lightgbm_weight"] for row in rows], [row["pr_auc"] for row in rows], marker="o", color="#0f766e")
    ax.axvline(best["lightgbm_weight"], color="#f97316", ls="--", label=f"Best LGB weight={best['lightgbm_weight']:.2f}")
    ax.set(xlabel="Engineered LightGBM ağırlığı", ylabel="Validation PR-AUC", title="Ensemble Weight Search")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "ensemble_weight_search.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_ensemble_experiment():
    data = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    forest, lightgbm, feature_report = _candidate_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Random Forest OOF calibration", flush=True)
    calibrated_rf, raw_rf, rf_oof = _fit_calibrated(forest, X_train, y_train, cv)
    print("Engineered LightGBM OOF calibration", flush=True)
    calibrated_lgb, raw_lgb, lgb_oof = _fit_calibrated(lightgbm, X_train, y_train, cv)

    X_weight, X_threshold, y_weight, y_threshold = train_test_split(
        X_val, y_val, test_size=.5, random_state=42, stratify=y_val,
    )
    rf_weight = calibrated_rf.predict_proba(X_weight)[:, 1]
    lgb_weight = calibrated_lgb.predict_proba(X_weight)[:, 1]
    weight_rows, selected_weight = _choose_weight(y_weight, rf_weight, lgb_weight)
    weight = selected_weight["lightgbm_weight"]
    ensemble = CalibratedSoftVotingEnsemble(calibrated_rf, calibrated_lgb, weight)

    threshold_proba = ensemble.predict_proba(X_threshold)[:, 1]
    thresholds = choose_thresholds(y_threshold, threshold_proba)
    threshold = float(thresholds["cost_optimal"]["threshold"])
    test_proba = ensemble.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(y_test, test_proba, "RF + Engineered LightGBM Ensemble", threshold=threshold)

    component_validation = {
        "random_forest_pr_auc": float(average_precision_score(y_weight, rf_weight)),
        "engineered_lightgbm_pr_auc": float(average_precision_score(y_weight, lgb_weight)),
        "ensemble_pr_auc": selected_weight["pr_auc"],
    }
    minimum_delta = .001
    promote = selected_weight["pr_auc"] >= max(component_validation["random_forest_pr_auc"], component_validation["engineered_lightgbm_pr_auc"]) + minimum_delta
    raw_test = (1 - weight) * raw_rf.predict_proba(X_test)[:, 1] + weight * raw_lgb.predict_proba(X_test)[:, 1]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "OOF calibration on train; weight on first validation half; threshold on second validation half; test reporting-only.",
        "weight_search": weight_rows,
        "selected_weights": selected_weight,
        "component_validation": component_validation,
        "promotion_minimum_pr_auc_delta": minimum_delta,
        "promoted_to_final": promote,
        "oof_train_pr_auc": {
            "random_forest": float(average_precision_score(y_train, rf_oof)),
            "engineered_lightgbm": float(average_precision_score(y_train, lgb_oof)),
        },
        "threshold_analysis": thresholds,
        "calibration_test": calibration_metrics(y_test, raw_test, test_proba),
        "test_metrics": test_metrics,
        "previous_final_test_metrics": feature_report["test_metrics"],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True); MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "ensemble_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(ensemble, MODELS_DIR / "ensemble_model.pkl", compress=3)
    if promote:
        joblib.dump(ensemble, MODELS_DIR / "best_model.pkl", compress=3)
        metadata = {
            "model_name": "Calibrated RF + Engineered LightGBM Ensemble",
            "threshold": threshold, "default_threshold": .5,
            "threshold_policy": "minimum validation cost with FN=2, FP=1",
            "trained_at": report["created_at"], "data_rows": len(data),
            "data_policy": "exact feature-and-target duplicates removed before split",
            "split_rows": {"train": len(X_train), "weight_validation": len(X_weight), "threshold_validation": len(X_threshold), "test": len(X_test)},
            "ensemble": {"random_forest_weight": 1 - weight, "engineered_lightgbm_weight": weight},
            "segment_policy": {
                "strategy": "fixed business bands aligned to validation-selected threshold",
                "codes": ["LOW", "MEDIUM", "HIGH"],
                "boundaries": [0.0, threshold / 2, threshold, 1.0],
            },
            "test_metrics": test_metrics,
        }
        (MODELS_DIR / "threshold.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_weight_search(weight_rows)
    print(f"Weights RF={1-weight:.2f}, LGB={weight:.2f}; promoted={promote}")
    print(f"Ensemble test PR-AUC={test_metrics['pr_auc']:.4f}; F1={test_metrics['f1_optimized']:.4f}")
    return report


if __name__ == "__main__":
    run_ensemble_experiment()
