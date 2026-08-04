"""Model metrics, threshold optimisation, and report visualisations."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def threshold_scores(y_true, y_proba) -> list[dict[str, float]]:
    """Calculate positive-class scores for thresholds from .05 through .94."""
    rows = []
    for threshold in np.arange(0.05, 0.95, 0.01):
        prediction = (np.asarray(y_proba) >= threshold).astype(int)
        rows.append({
            "threshold": round(float(threshold), 2),
            "f1": float(f1_score(y_true, prediction, zero_division=0)),
            "precision": float(precision_score(y_true, prediction, zero_division=0)),
            "recall": float(recall_score(y_true, prediction, zero_division=0)),
        })
    return rows


def find_optimal_threshold(y_true, y_proba) -> float:
    """Return the first threshold yielding the maximum positive-class F1."""
    rows = threshold_scores(y_true, y_proba)
    return max(rows, key=lambda row: row["f1"])["threshold"]


def evaluate_model(y_true, y_proba, model_name: str, threshold: float | None = None) -> dict:
    """Return probability and classification metrics at default/selected thresholds."""
    selected = threshold if threshold is not None else find_optimal_threshold(y_true, y_proba)
    default_pred = (np.asarray(y_proba) >= 0.5).astype(int)
    selected_pred = (np.asarray(y_proba) >= selected).astype(int)
    report = classification_report(y_true, selected_pred, output_dict=True, zero_division=0)
    return {
        "model": model_name,
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "threshold_default": 0.5,
        "threshold_optimized": float(selected),
        "f1_default": float(f1_score(y_true, default_pred, zero_division=0)),
        "f1_optimized": float(f1_score(y_true, selected_pred, zero_division=0)),
        "precision_optimized": float(precision_score(y_true, selected_pred, zero_division=0)),
        "recall_optimized": float(recall_score(y_true, selected_pred, zero_division=0)),
        "accuracy_optimized": float(report["accuracy"]),
        "report_optimized": report,
        "confusion_matrix": confusion_matrix(y_true, selected_pred).tolist(),
    }


def _save(fig, filename: str) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_proba, threshold: float, model_name: str) -> None:
    prediction = (np.asarray(y_proba) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, prediction)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Bye", "Buy"], yticklabels=["Bye", "Buy"])
    ax.set(xlabel="Tahmin", ylabel="Gerçek",
           title=f"Karışıklık Matrisi — {model_name} (eşik={threshold:.2f})")
    _save(fig, "confusion_matrix.png")


def plot_pr_curve(y_true, y_proba, model_name: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    score = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#10b981", lw=2, label=f"{model_name} (PR-AUC={score:.3f})")
    ax.axhline(np.mean(y_true), color="#94a3b8", ls="--", label="Pozitif sınıf oranı")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Eğrisi")
    ax.grid(alpha=.2); ax.legend()
    _save(fig, "pr_curve.png")


def plot_roc_curve(y_true, y_proba, model_name: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    score = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#7c3aed", lw=2, label=f"{model_name} (ROC-AUC={score:.3f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", ls="--")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Eğrisi")
    ax.grid(alpha=.2); ax.legend()
    _save(fig, "roc_curve.png")


def plot_threshold_optimization(y_true, y_proba, model_name: str) -> None:
    rows = threshold_scores(y_true, y_proba)
    optimal = max(rows, key=lambda row: row["f1"])
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric, color in [("f1", "#10b981"), ("precision", "#f97316"), ("recall", "#2563eb")]:
        ax.plot([r["threshold"] for r in rows], [r[metric] for r in rows], label=metric.title(), color=color)
    ax.axvline(optimal["threshold"], color="#475569", ls="--", label=f"Optimal {optimal['threshold']:.2f}")
    ax.set(xlabel="Eşik", ylabel="Skor", title=f"Eşik Optimizasyonu — {model_name}")
    ax.grid(alpha=.2); ax.legend()
    _save(fig, "threshold_optimization.png")
