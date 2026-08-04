"""Evaluate and visualise probability segments on the untouched test split."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_data
from .predict import load_artifacts, predict_sessions
from .segments import build_segments
from .train import split_data

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"


def analyse_segments(y_true, scored: pd.DataFrame, threshold: float) -> dict:
    """Summarise volume, observed conversion, and buyer capture by segment."""
    frame = scored.copy()
    frame["actual_purchase"] = np.asarray(y_true, dtype=int)
    definitions = build_segments(threshold)
    total_buyers = max(int(frame["actual_purchase"].sum()), 1)
    rows = []
    for definition in definitions:
        subset = frame[frame["segment"] == definition.code]
        buyers = int(subset["actual_purchase"].sum())
        rows.append({
            **definition.to_dict(),
            "upper": min(definition.upper, 1.0),
            "sessions": int(len(subset)),
            "session_share": float(len(subset) / len(frame)),
            "buyers": buyers,
            "buyer_capture": float(buyers / total_buyers),
            "observed_conversion_rate": float(subset["actual_purchase"].mean()) if len(subset) else 0.0,
            "mean_predicted_probability": float(subset["purchase_probability"].mean()) if len(subset) else 0.0,
        })
    return {
        "model_output": "calibrated predict_proba[:, 1]",
        "decision_threshold": float(threshold),
        "evaluated_rows": int(len(frame)),
        "actual_buyers": int(frame["actual_purchase"].sum()),
        "segments": rows,
    }


def plot_segment_performance(report: dict) -> None:
    """Save segment volume and conversion-quality diagnostics headlessly."""
    rows = report["segments"]
    labels = [row["label"] for row in rows]
    colors = [row["color"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    bars = axes[0].bar(x, [row["sessions"] for row in rows], color=colors)
    axes[0].bar_label(bars, padding=3, fontsize=9)
    axes[0].set(title="Segment Hacmi", ylabel="Oturum sayısı", xticks=x, xticklabels=labels)

    width = .36
    observed = [row["observed_conversion_rate"] for row in rows]
    predicted = [row["mean_predicted_probability"] for row in rows]
    axes[1].bar(x - width / 2, observed, width, label="Gerçek dönüşüm", color=colors)
    axes[1].bar(x + width / 2, predicted, width, label="Ortalama tahmin", color="#cbd5e1")
    axes[1].set(title="Segment Kalitesi ve Kalibrasyon", ylabel="Oran", ylim=(0, 1),
                xticks=x, xticklabels=labels)
    axes[1].legend()
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=.2)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "segment_performance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    """Score the fixed test split, persist the segment report, and draw its chart."""
    data = load_data()
    _, _, X_test, _, _, y_test = split_data(data)
    model, metadata = load_artifacts()
    scored = predict_sessions(X_test, model, metadata)
    report = analyse_segments(y_test, scored, float(metadata["threshold"]))
    report.update({
        "model_name": metadata["model_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_note": "Segment sınırları model eğitiminden sonra tanımlandı; test seti yalnızca raporlama için kullanıldı.",
    })
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "segment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_segment_performance(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
