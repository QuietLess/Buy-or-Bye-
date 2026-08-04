"""Probability-based customer segments and their recommended business actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ProbabilitySegment:
    """A left-closed probability interval with an operational recommendation."""

    code: str
    label: str
    lower: float
    upper: float
    action: str
    objective: str
    color: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_segments(decision_threshold: float) -> tuple[ProbabilitySegment, ...]:
    """Build stable bands aligned with the calibrated business threshold.

    The medium/high boundary is the validation-selected operating threshold;
    its midpoint separates low and medium intent. Three bands keep the output
    operationally clear without changing the trained model or its threshold.
    """
    threshold = float(decision_threshold)
    if not 0.10 <= threshold < 0.50:
        raise ValueError("Segmentasyon için karar eşiği [0.10, 0.50) aralığında olmalı.")
    midpoint = threshold / 2
    return (
        ProbabilitySegment(
            "LOW", "Düşük niyet", 0.0, midpoint,
            "Ücretli teşvik verme; düşük maliyetli içerik ve yeniden hedefleme ile ilgiyi geliştir.",
            "Gereksiz kampanya maliyetini önleyip ilgiyi geliştirmek", "#64748b",
        ),
        ProbabilitySegment(
            "MEDIUM", "Değerlendirme aşaması", midpoint, threshold,
            "Sosyal kanıt, ürün karşılaştırma ve stok/kargo bilgisini görünür kıl.",
            "Kararsızlığı azaltıp yüksek niyete taşımak", "#eab308",
        ),
        ProbabilitySegment(
            "HIGH", "Yüksek niyet", threshold, 1.0000001,
            "Checkout desteği ve kişiselleştirilmiş hatırlatma sun; teşviki kontrollü test ederek marjı koru.",
            "Satın alma sürtünmesini azaltıp dönüşümü tamamlamak", "#16a34a",
        ),
    )


def segment_probability(probability: float, decision_threshold: float) -> dict:
    """Return the segment and business action for one calibrated probability."""
    probability = float(probability)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Olasılık 0 ile 1 arasında sonlu bir değer olmalı.")
    for segment in build_segments(decision_threshold):
        if segment.lower <= probability < segment.upper:
            result = segment.to_dict()
            result["probability"] = probability
            return result
    raise RuntimeError("Olasılık bir segmente atanamadı.")


def segment_probabilities(probabilities, decision_threshold: float) -> list[dict]:
    """Assign multiple probabilities while preserving input order."""
    return [segment_probability(value, decision_threshold) for value in probabilities]
