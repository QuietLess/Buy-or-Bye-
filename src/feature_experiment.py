"""Compare baseline and engineered features, tune candidates, and promote by CV."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

from .data import load_data
from .evaluate import evaluate_model
from .features import ENGINEERED_NUMERIC_FEATURES, build_engineered_pipeline
from .train import MODELS_DIR, REPORTS_DIR, build_lightgbm, build_random_forest, split_data
from .tune import (
    RANDOM_STATE, _fit_and_calibrate, calibration_metrics, choose_thresholds,
    multi_seed_stability, pagevalues_ablation, plot_calibration, plot_stability,
    temporal_proxy,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_tuned_baselines():
    report = json.loads((REPORTS_DIR / "tuning_report.json").read_text(encoding="utf-8"))
    by_name = {row["model"]: row for row in report["cv_results"]}
    from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    forest = build_random_forest(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    forest.set_params(**by_name["Random Forest"]["best_params"])
    lightgbm = build_lightgbm(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    lightgbm.set_params(**by_name["LightGBM"]["best_params"])
    return report, forest, lightgbm


def _score(model, X, y, cv):
    values = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=1)
    return {"mean": float(values.mean()), "std": float(values.std()), "folds": values.tolist()}


def run_feature_experiment(quick: bool = False):
    data = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    baseline_report, baseline_forest, baseline_lightgbm = _load_tuned_baselines()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    engineered_forest = build_engineered_pipeline(baseline_forest)
    engineered_lightgbm = build_engineered_pipeline(baseline_lightgbm)

    print("Sabit parametrelerle feature etkisi", flush=True)
    fixed = {
        "baseline_random_forest": next(row for row in baseline_report["cv_results"] if row["model"] == "Random Forest")["cv_pr_auc_mean"],
        "baseline_lightgbm": next(row for row in baseline_report["cv_results"] if row["model"] == "LightGBM")["cv_pr_auc_mean"],
        "engineered_random_forest": _score(engineered_forest, X_train, y_train, cv),
        "engineered_lightgbm": _score(engineered_lightgbm, X_train, y_train, cv),
    }

    common = {"scoring": "average_precision", "cv": cv, "n_jobs": 1, "refit": True, "return_train_score": True, "random_state": RANDOM_STATE}
    searches = {
        "Engineered Random Forest": RandomizedSearchCV(
            engineered_forest,
            {
                "classifier__n_estimators": [250, 350, 500],
                "classifier__max_depth": [12, 18, None],
                "classifier__min_samples_leaf": [4, 8, 12, 16],
                "classifier__min_samples_split": [5, 10, 20],
                "classifier__max_features": [0.3, 0.5, 0.7],
                "classifier__class_weight": ["balanced", "balanced_subsample"],
            },
            n_iter=3 if quick else 8, **common,
        ),
        "Engineered LightGBM": RandomizedSearchCV(
            engineered_lightgbm,
            {
                "classifier__n_estimators": [150, 200, 300, 450],
                "classifier__learning_rate": [0.02, 0.03, 0.05],
                "classifier__max_depth": [4, 5, 6, 7],
                "classifier__num_leaves": [15, 23, 31, 47],
                "classifier__min_child_samples": [20, 35, 50, 80],
                "classifier__subsample": [0.7, 0.85, 1.0],
                "classifier__subsample_freq": [1],
                "classifier__colsample_bytree": [0.7, 0.85, 1.0],
                "classifier__reg_alpha": [0.0, 0.1, 0.3, 0.7],
                "classifier__reg_lambda": [0.0, 0.2, 0.7, 1.5],
            },
            n_iter=4 if quick else 12, **common,
        ),
    }
    tuned_rows = []
    for name, search in searches.items():
        print(f"Feature tuning: {name}", flush=True)
        search.fit(X_train, y_train)
        index = search.best_index_
        tuned_rows.append({
            "model": name,
            "cv_pr_auc_mean": float(search.cv_results_["mean_test_score"][index]),
            "cv_pr_auc_std": float(search.cv_results_["std_test_score"][index]),
            "cv_train_pr_auc_mean": float(search.cv_results_["mean_train_score"][index]),
            "best_params": search.best_params_,
        })
    winner_row = max(tuned_rows, key=lambda row: row["cv_pr_auc_mean"])
    winner_name = winner_row["model"]
    winner_family = "LightGBM" if "LightGBM" in winner_name else "Random Forest"
    winner_search = searches[winner_name]
    calibrated, raw_model, best_iteration, _ = _fit_and_calibrate(
        winner_search.best_estimator_, winner_family, X_train, y_train, X_val, y_val, cv,
    )
    raw_val, calibrated_val = raw_model.predict_proba(X_val)[:, 1], calibrated.predict_proba(X_val)[:, 1]
    raw_test, calibrated_test = raw_model.predict_proba(X_test)[:, 1], calibrated.predict_proba(X_test)[:, 1]
    thresholds = choose_thresholds(y_val, calibrated_val)
    threshold = float(thresholds["cost_optimal"]["threshold"])
    test_metrics = evaluate_model(y_test, calibrated_test, winner_name, threshold=threshold)
    baseline_cv = max(row["cv_pr_auc_mean"] for row in baseline_report["cv_results"])
    promote = winner_row["cv_pr_auc_mean"] > baseline_cv

    print("Engineered robustness analizleri", flush=True)
    ablation = pagevalues_ablation(winner_search.best_estimator_, X_train, y_train, cv)
    stability = multi_seed_stability(winner_search.best_estimator_, data)
    temporal = temporal_proxy(winner_search.best_estimator_, data)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Promote only when engineered 5-fold train CV PR-AUC exceeds baseline; test is reporting-only.",
        "engineered_features": ENGINEERED_NUMERIC_FEATURES,
        "fixed_parameter_comparison": fixed,
        "tuned_engineered_results": tuned_rows,
        "baseline_best_cv_pr_auc": baseline_cv,
        "engineered_winner": winner_name,
        "engineered_winner_cv_pr_auc": winner_row["cv_pr_auc_mean"],
        "promoted_to_final": promote,
        "best_iteration_early_stopping": best_iteration,
        "threshold_analysis": thresholds,
        "calibration": {
            "validation": calibration_metrics(y_val, raw_val, calibrated_val),
            "test": calibration_metrics(y_test, raw_test, calibrated_test),
        },
        "pagevalues_ablation": ablation,
        "multi_seed_stability": stability,
        "temporal_proxy": temporal,
        "test_metrics": test_metrics,
        "previous_final_test_metrics": baseline_report["test_metrics"],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "feature_engineering_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if promote:
        joblib.dump(calibrated, MODELS_DIR / "best_model.pkl", compress=3)
        metadata = {
            "model_name": f"Calibrated {winner_name}", "threshold": threshold,
            "default_threshold": 0.5, "threshold_policy": "minimum validation cost with FN=2, FP=1",
            "trained_at": report["created_at"], "data_rows": len(data),
            "split_rows": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
            "feature_engineering": True, "test_metrics": test_metrics,
        }
        (MODELS_DIR / "threshold.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        plot_calibration(y_test, raw_test, calibrated_test); plot_stability(stability)
    print(f"Feature CV: {winner_row['cv_pr_auc_mean']:.4f}; baseline CV: {baseline_cv:.4f}; promoted={promote}")
    print(f"Engineered test PR-AUC={test_metrics['pr_auc']:.4f}; F1={test_metrics['f1_optimized']:.4f}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_feature_experiment(quick=args.quick)


if __name__ == "__main__":
    main()
