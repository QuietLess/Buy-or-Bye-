"""Train, compare, select, evaluate, and persist Buy or Bye models."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import TARGET, get_feature_groups, load_data
from .evaluate import (
    evaluate_model, find_optimal_threshold, plot_confusion_matrix,
    plot_pr_curve, plot_roc_curve, plot_threshold_optimization,
)
from .preprocess import build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def build_logistic_regression(numeric_cols, categorical_cols) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])


def build_random_forest(numeric_cols, categorical_cols) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight="balanced",
            random_state=42, n_jobs=1,
        )),
    ])


def build_lightgbm(numeric_cols, categorical_cols) -> Pipeline:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError("LightGBM modeli için `pip install lightgbm` gereklidir.") from exc
    return Pipeline([
        ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
        ("classifier", LGBMClassifier(
            n_estimators=300, learning_rate=.05, max_depth=7, num_leaves=31,
            min_child_samples=20, subsample=.8, colsample_bytree=.8,
            class_weight="balanced", random_state=42, n_jobs=1, verbosity=-1,
            force_col_wise=True,
        )),
    ])


def split_data(df: pd.DataFrame):
    """Create deterministic stratified 70/15/15 train, validation, and test sets."""
    return _split_named(df)


def _split_named(df: pd.DataFrame):
    X, y = df.drop(columns=TARGET), df[TARGET].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=.50, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _train_candidate_worker(name, df, numeric, categorical, artifact_path, queue) -> None:
    """Fit one candidate in an isolated process to avoid macOS OpenMP conflicts."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    builders = {
        "LightGBM": build_lightgbm,
        "Logistic Regression": build_logistic_regression,
        "Random Forest": build_random_forest,
    }
    X_train, X_val, X_test, y_train, y_val, y_test = _split_named(df)
    try:
        model = builders[name](numeric, categorical)
        model.fit(X_train, y_train)
        joblib.dump(model, artifact_path, compress=3)
        queue.put({
            "name": name,
            "validation_proba": model.predict_proba(X_val)[:, 1],
            "test_proba": model.predict_proba(X_test)[:, 1],
        })
    except Exception as exc:
        queue.put({"name": name, "error": repr(exc)})


def _fit_candidates_isolated(df, numeric, categorical):
    candidate_dir = MODELS_DIR / ".candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    outputs = {}
    for name in ["Logistic Regression", "Random Forest", "LightGBM"]:
        print(f"Eğitiliyor: {name}", flush=True)
        artifact = candidate_dir / f"{name.lower().replace(' ', '_')}.pkl"
        queue = context.Queue()
        process = context.Process(
            target=_train_candidate_worker,
            args=(name, df, numeric, categorical, artifact, queue),
        )
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"{name} eğitimi native süreçte başarısız: {process.exitcode}")
        payload = queue.get()
        if "error" in payload:
            raise RuntimeError(f"{name} eğitimi başarısız: {payload.get('error', process.exitcode)}")
        payload["artifact"] = artifact
        outputs[name] = payload
    return outputs


def _explanation_worker(model_path, X_train, model_name, generate_shap, queue) -> None:
    """Generate model-native/SHAP figures in a clean native-library process."""
    try:
        model = joblib.load(model_path)
        from .explain import plot_feature_importance, plot_shap_summary
        plot_feature_importance(model)
        if generate_shap:
            plot_shap_summary(model, X_train, model_name)
        queue.put({"ok": True})
    except Exception as exc:
        queue.put({"ok": False, "error": repr(exc)})


def _generate_explanations_isolated(model_path, X_train, model_name, generate_shap) -> None:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_explanation_worker,
        args=(model_path, X_train, model_name, generate_shap, queue),
    )
    process.start(); process.join()
    if process.exitcode != 0:
        print(f"Açıklama süreci native seviyede sonlandı: {process.exitcode}")
        return
    payload = queue.get()
    if not payload["ok"]:
        print(f"SHAP/native importance üretilemedi: {payload['error']}")


def _report_worker(y_val, val_proba, y_test, test_proba, threshold, model_name, queue) -> None:
    try:
        plot_confusion_matrix(y_test, test_proba, threshold, model_name)
        plot_pr_curve(y_test, test_proba, model_name)
        plot_roc_curve(y_test, test_proba, model_name)
        plot_threshold_optimization(y_val, val_proba, model_name)
        queue.put({"ok": True})
    except Exception as exc:
        queue.put({"ok": False, "error": repr(exc)})


def _generate_reports_isolated(y_val, val_proba, y_test, test_proba, threshold, model_name) -> None:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_report_worker,
        args=(y_val, val_proba, y_test, test_proba, threshold, model_name, queue),
    )
    process.start(); process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"Rapor görselleri üretilemedi: {process.exitcode}")
    payload = queue.get()
    if not payload["ok"]:
        raise RuntimeError(f"Rapor görselleri üretilemedi: {payload['error']}")


def train_all(df: pd.DataFrame | None = None, generate_explanations: bool = True) -> dict:
    """Run the complete experiment and write reproducible model/report artifacts."""
    df = load_data() if df is None else df
    groups = get_feature_groups(df)
    X_train, X_val, X_test, y_train, y_val, y_test = _split_named(df)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    candidate_outputs = _fit_candidates_isolated(df, groups["numeric"], groups["categorical"])
    validation_rows, probabilities = [], {}
    for name, output in candidate_outputs.items():
        proba = output["validation_proba"]
        probabilities[name] = proba
        validation_rows.append(evaluate_model(y_val, proba, name))

    # PR-AUC is the primary selection metric for this imbalanced target.
    best_validation = max(validation_rows, key=lambda row: (row["pr_auc"], row["f1_optimized"]))
    best_name = best_validation["model"]
    threshold = find_optimal_threshold(y_val, probabilities[best_name])

    test_proba = candidate_outputs[best_name]["test_proba"]
    test_metrics = evaluate_model(y_test, test_proba, best_name, threshold=threshold)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_outputs[best_name]["artifact"], MODELS_DIR / "best_model.pkl")

    metadata = {
        "model_name": best_name,
        "threshold": threshold,
        "default_threshold": .5,
        "selection_metric": "validation_pr_auc",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_rows": len(df),
        "split_rows": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "validation_metrics": best_validation,
        "test_metrics": test_metrics,
    }
    with (MODELS_DIR / "threshold.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    with (REPORTS_DIR / "model_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump({"validation": validation_rows, "test": test_metrics}, handle, ensure_ascii=False, indent=2)

    print("Değerlendirme görselleri üretiliyor", flush=True)
    _generate_reports_isolated(
        y_val, probabilities[best_name], y_test, test_proba, threshold, best_name,
    )
    print("Açıklanabilirlik görselleri üretiliyor", flush=True)
    _generate_explanations_isolated(
        MODELS_DIR / "best_model.pkl", X_train, best_name, generate_explanations,
    )

    print(f"En iyi model: {best_name} | validation PR-AUC={best_validation['pr_auc']:.4f}")
    print(f"Test ROC-AUC={test_metrics['roc_auc']:.4f} | PR-AUC={test_metrics['pr_auc']:.4f} | F1={test_metrics['f1_optimized']:.4f}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Alternatif CSV yolu")
    parser.add_argument("--skip-shap", action="store_true", help="SHAP özetini atla")
    args = parser.parse_args()
    train_all(load_data(args.data) if args.data else None, generate_explanations=not args.skip_shap)


if __name__ == "__main__":
    main()
