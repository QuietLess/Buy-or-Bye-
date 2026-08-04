"""Cross-validated tuning, calibration, robustness checks, and final training."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import early_stopping
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict,
    cross_val_score, train_test_split,
)

from .calibration import SigmoidCalibratedModel
from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, load_data
from .evaluate import evaluate_model, threshold_scores
from .preprocess import build_preprocessor
from .train import (
    MODELS_DIR, REPORTS_DIR, build_lightgbm, build_logistic_regression,
    build_random_forest, split_data,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
CV_SPLITS = 5
RANDOM_STATE = 42


def _searches(n_jobs: int = 1, quick: bool = False):
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    common = {"scoring": "average_precision", "cv": cv, "n_jobs": n_jobs, "refit": True, "return_train_score": True}
    logistic = GridSearchCV(
        build_logistic_regression(NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        {
            "classifier__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "classifier__solver": ["liblinear"],
            "classifier__penalty": ["l1", "l2"],
        },
        **common,
    )
    forest = RandomizedSearchCV(
        build_random_forest(NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        {
            "classifier__n_estimators": [150, 250, 350],
            "classifier__max_depth": [8, 12, 16, 20, None],
            "classifier__min_samples_leaf": [1, 2, 4, 8, 12],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__max_features": ["sqrt", 0.5, 0.8],
            "classifier__class_weight": ["balanced", "balanced_subsample"],
        },
        n_iter=3 if quick else 8, random_state=RANDOM_STATE, **common,
    )
    lightgbm = RandomizedSearchCV(
        build_lightgbm(NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        {
            "classifier__n_estimators": [200, 350, 500, 700, 900],
            "classifier__learning_rate": [0.02, 0.03, 0.05, 0.08],
            "classifier__max_depth": [4, 5, 6, 7, 8, -1],
            "classifier__num_leaves": [15, 23, 31, 47, 63],
            "classifier__min_child_samples": [10, 20, 35, 50, 80],
            "classifier__subsample": [0.7, 0.85, 1.0],
            "classifier__subsample_freq": [1],
            "classifier__colsample_bytree": [0.65, 0.8, 0.95, 1.0],
            "classifier__reg_alpha": [0.0, 0.05, 0.2, 0.5, 1.0],
            "classifier__reg_lambda": [0.0, 0.1, 0.5, 1.0, 2.0],
        },
        n_iter=5 if quick else 16, random_state=RANDOM_STATE, **common,
    )
    return {"Logistic Regression": logistic, "Random Forest": forest, "LightGBM": lightgbm}, cv


def tune_candidates(X_train, y_train, quick: bool = False):
    searches, cv = _searches(quick=quick)
    summary = []
    for name, search in searches.items():
        print(f"5-fold tuning: {name}", flush=True)
        search.fit(X_train, y_train)
        index = search.best_index_
        summary.append({
            "model": name,
            "cv_pr_auc_mean": float(search.cv_results_["mean_test_score"][index]),
            "cv_pr_auc_std": float(search.cv_results_["std_test_score"][index]),
            "cv_train_pr_auc_mean": float(search.cv_results_["mean_train_score"][index]),
            "best_params": search.best_params_,
        })
    winner = max(summary, key=lambda row: row["cv_pr_auc_mean"])["model"]
    return searches, summary, winner, cv


def _lightgbm_best_iteration(best_pipeline, X_train, y_train, X_val, y_val) -> int:
    feature_engineer = clone(best_pipeline.named_steps["feature_engineer"]) if "feature_engineer" in best_pipeline.named_steps else None
    preprocessor = clone(best_pipeline.named_steps["preprocessor"])
    classifier = clone(best_pipeline.named_steps["classifier"])
    classifier.set_params(n_estimators=max(1500, classifier.get_params()["n_estimators"]))
    engineered_train = feature_engineer.fit_transform(X_train, y_train) if feature_engineer is not None else X_train
    engineered_val = feature_engineer.transform(X_val) if feature_engineer is not None else X_val
    transformed_train = preprocessor.fit_transform(engineered_train)
    transformed_val = preprocessor.transform(engineered_val)
    classifier.fit(
        transformed_train, y_train, eval_set=[(transformed_val, y_val)],
        eval_metric="average_precision", callbacks=[early_stopping(75, verbose=False)],
    )
    return int(classifier.best_iteration_ or classifier.get_params()["n_estimators"])


def _fit_and_calibrate(best_pipeline, winner, X_train, y_train, X_val, y_val, cv):
    selected = clone(best_pipeline)
    best_iteration = None
    if winner == "LightGBM":
        best_iteration = _lightgbm_best_iteration(selected, X_train, y_train, X_val, y_val)
        selected.set_params(classifier__n_estimators=best_iteration)
    print("OOF sigmoid calibration", flush=True)
    oof_proba = cross_val_predict(selected, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    selected.fit(X_train, y_train)
    calibrated = SigmoidCalibratedModel(selected).fit_calibrator(oof_proba, y_train)
    return calibrated, selected, best_iteration, oof_proba


def choose_thresholds(y_true, probabilities, false_positive_cost=1.0, false_negative_cost=2.0):
    rows = threshold_scores(y_true, probabilities)
    for row in rows:
        pred = (np.asarray(probabilities) >= row["threshold"]).astype(int)
        fp = int(((pred == 1) & (np.asarray(y_true) == 0)).sum())
        fn = int(((pred == 0) & (np.asarray(y_true) == 1)).sum())
        row.update({"fp": fp, "fn": fn, "business_cost": fp * false_positive_cost + fn * false_negative_cost})
    f1_row = max(rows, key=lambda row: row["f1"])
    cost_row = min(rows, key=lambda row: (row["business_cost"], -row["f1"]))
    return {"f1_optimal": f1_row, "cost_optimal": cost_row, "costs": {"false_positive": false_positive_cost, "false_negative": false_negative_cost}}


def calibration_metrics(y_true, raw_proba, calibrated_proba):
    return {
        "raw_brier": float(brier_score_loss(y_true, raw_proba)),
        "calibrated_brier": float(brier_score_loss(y_true, calibrated_proba)),
        "raw_log_loss": float(log_loss(y_true, raw_proba)),
        "calibrated_log_loss": float(log_loss(y_true, calibrated_proba)),
    }


def pagevalues_ablation(best_pipeline, X_train, y_train, cv):
    full = cross_val_score(clone(best_pipeline), X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)
    if "feature_engineer" in best_pipeline.named_steps:
        from .features import ALL_NUMERIC_FEATURES
        numeric_without = [column for column in ALL_NUMERIC_FEATURES if column != "PageValues"]
    else:
        numeric_without = [column for column in NUMERIC_FEATURES if column != "PageValues"]
    without = clone(best_pipeline)
    without.set_params(preprocessor=build_preprocessor(numeric_without, CATEGORICAL_FEATURES))
    reduced_X = X_train.drop(columns="PageValues")
    reduced = cross_val_score(without, reduced_X, y_train, cv=cv, scoring="average_precision", n_jobs=1)
    return {
        "with_pagevalues_pr_auc_mean": float(full.mean()),
        "with_pagevalues_pr_auc_std": float(full.std()),
        "without_pagevalues_pr_auc_mean": float(reduced.mean()),
        "without_pagevalues_pr_auc_std": float(reduced.std()),
        "pr_auc_drop": float(full.mean() - reduced.mean()),
    }


def multi_seed_stability(best_pipeline, df, seeds=(7, 21, 42, 73, 101)):
    X, y = df.drop(columns=TARGET), df[TARGET].astype(int)
    rows = []
    for seed in seeds:
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y,
        )
        model = clone(best_pipeline).fit(X_train, y_train)
        proba = model.predict_proba(X_holdout)[:, 1]
        rows.append({"seed": seed, "pr_auc": float(average_precision_score(y_holdout, proba)), "roc_auc": float(roc_auc_score(y_holdout, proba))})
    return {
        "runs": rows,
        "pr_auc_mean": float(np.mean([row["pr_auc"] for row in rows])),
        "pr_auc_std": float(np.std([row["pr_auc"] for row in rows])),
        "roc_auc_mean": float(np.mean([row["roc_auc"] for row in rows])),
        "roc_auc_std": float(np.std([row["roc_auc"] for row in rows])),
    }


def temporal_proxy(best_pipeline, df):
    month_order = {"Feb": 2, "Mar": 3, "May": 5, "June": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    order = df["Month"].map(month_order)
    train_mask, holdout_mask = order <= 10, order >= 11
    X_train, y_train = df.loc[train_mask].drop(columns=TARGET), df.loc[train_mask, TARGET].astype(int)
    X_holdout, y_holdout = df.loc[holdout_mask].drop(columns=TARGET), df.loc[holdout_mask, TARGET].astype(int)
    model = clone(best_pipeline).fit(X_train, y_train)
    proba = model.predict_proba(X_holdout)[:, 1]
    return {
        "definition": "Train Feb-Oct; holdout Nov-Dec (month proxy, no timestamp/year available)",
        "train_rows": len(X_train), "holdout_rows": len(X_holdout),
        "holdout_positive_rate": float(y_holdout.mean()),
        "pr_auc": float(average_precision_score(y_holdout, proba)),
        "roc_auc": float(roc_auc_score(y_holdout, proba)),
    }


def plot_calibration(y_true, raw_proba, calibrated_proba) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, probabilities, color in [("Raw", raw_proba, "#f97316"), ("OOF sigmoid", calibrated_proba, "#0f766e")]:
        observed, predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=label, color=color)
    ax.plot([0, 1], [0, 1], "--", color="#64748b", label="Perfect")
    ax.set(xlabel="Tahmin edilen olasılık", ylabel="Gözlenen oran", title="Calibration Curve")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "calibration_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_stability(stability) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(stability["runs"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["seed"].astype(str), frame["pr_auc"], marker="o", label="PR-AUC", color="#0f766e")
    ax.plot(frame["seed"].astype(str), frame["roc_auc"], marker="o", label="ROC-AUC", color="#7c3aed")
    ax.set(xlabel="Random seed", ylabel="Skor", ylim=(0.5, 1), title="Multi-seed Holdout Stability")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "stability_analysis.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_tuning(quick: bool = False):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    searches, cv_summary, winner, cv = tune_candidates(X_train, y_train, quick=quick)
    best_search = searches[winner]
    calibrated, raw_model, best_iteration, _ = _fit_and_calibrate(
        best_search.best_estimator_, winner, X_train, y_train, X_val, y_val, cv,
    )
    raw_val = raw_model.predict_proba(X_val)[:, 1]
    calibrated_val = calibrated.predict_proba(X_val)[:, 1]
    raw_test = raw_model.predict_proba(X_test)[:, 1]
    calibrated_test = calibrated.predict_proba(X_test)[:, 1]
    thresholds = choose_thresholds(y_val, calibrated_val)
    selected_threshold = float(thresholds["cost_optimal"]["threshold"])
    test_metrics = evaluate_model(y_test, calibrated_test, winner, threshold=selected_threshold)

    print("Robustness analizleri", flush=True)
    ablation = pagevalues_ablation(best_search.best_estimator_, X_train, y_train, cv)
    stability = multi_seed_stability(best_search.best_estimator_, df)
    temporal = temporal_proxy(best_search.best_estimator_, df)
    calibration = {
        "validation": calibration_metrics(y_val, raw_val, calibrated_val),
        "test": calibration_metrics(y_test, raw_test, calibrated_test),
    }
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "70/15/15 stratified holdout; tuning only on train with 5-fold CV",
        "model_selection_metric": "mean CV PR-AUC",
        "selected_model": winner,
        "cv_results": cv_summary,
        "best_iteration_early_stopping": best_iteration,
        "threshold_analysis": thresholds,
        "selected_threshold_policy": "minimum validation cost with FN=2, FP=1",
        "calibration": calibration,
        "pagevalues_ablation": ablation,
        "multi_seed_stability": stability,
        "temporal_proxy": temporal,
        "external_validation": "Not available: dataset contains one unidentified site only.",
        "test_metrics": test_metrics,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True); REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, MODELS_DIR / "best_model.pkl", compress=3)
    metadata = {
        "model_name": f"Calibrated {winner}",
        "threshold": selected_threshold,
        "default_threshold": 0.5,
        "threshold_policy": report["selected_threshold_policy"],
        "trained_at": report["trained_at"],
        "data_rows": len(df),
        "split_rows": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "test_metrics": test_metrics,
    }
    (MODELS_DIR / "threshold.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "tuning_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_calibration(y_test, raw_test, calibrated_test); plot_stability(stability)
    print(f"Seçilen model: {winner}; test PR-AUC={test_metrics['pr_auc']:.4f}; F1={test_metrics['f1_optimized']:.4f}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="CI/smoke test için küçük arama")
    args = parser.parse_args()
    run_tuning(quick=args.quick)


if __name__ == "__main__":
    main()
