"""Dataset acquisition, validation, and loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_shoppers_intention.csv"

NUMERIC_FEATURES = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
CATEGORICAL_FEATURES = [
    "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend", "Month",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Revenue"
EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET]


def _normalise_booleans(series: pd.Series, name: str) -> pd.Series:
    """Convert common boolean representations and reject unknown values."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {
        True: True, False: False, 1: True, 0: False,
        "True": True, "False": False, "TRUE": True, "FALSE": False,
        "1": True, "0": False,
    }
    converted = series.map(mapping)
    if converted.isna().any():
        values = sorted(map(str, series[converted.isna()].dropna().unique()))
        raise ValueError(f"{name} sütununda geçersiz boolean değerleri var: {values}")
    return converted.astype(bool)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and return a consistently typed defensive copy."""
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Veri setinde eksik sütunlar var: {missing}")
    clean = df.loc[:, EXPECTED_COLUMNS].copy()
    for column in NUMERIC_FEATURES + ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        clean[column] = pd.to_numeric(clean[column], errors="raise")
    clean["Weekend"] = _normalise_booleans(clean["Weekend"], "Weekend")
    clean[TARGET] = _normalise_booleans(clean[TARGET], TARGET)
    if clean.empty:
        raise ValueError("Veri seti boş.")
    return clean


def deduplicate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact feature-and-target duplicates before any data split."""
    return df.drop_duplicates(subset=EXPECTED_COLUMNS, keep="first").reset_index(drop=True)


def download_data() -> pd.DataFrame:
    """Download UCI dataset 468 and persist it as the canonical raw CSV."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "Veri indirmek için `pip install ucimlrepo` komutunu çalıştırın."
        ) from exc
    dataset = fetch_ucirepo(id=468)
    frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    frame = validate_data(frame)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False)
    return deduplicate_sessions(frame)


def load_data(path: str | Path | None = None, download_if_missing: bool = True) -> pd.DataFrame:
    """Load and validate the project dataset, downloading it when requested."""
    csv_path = Path(path) if path else DATA_PATH
    if not csv_path.exists():
        if path is not None or not download_if_missing:
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {csv_path}")
        return download_data()
    return deduplicate_sessions(validate_data(pd.read_csv(csv_path)))


def get_feature_groups(df: pd.DataFrame | None = None) -> dict[str, list[str] | str]:
    """Return the stable feature groups used by preprocessing and UI."""
    if df is not None:
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Beklenen sütunlar bulunamadı: {sorted(missing)}")
    return {
        "numeric": NUMERIC_FEATURES.copy(),
        "categorical": CATEGORICAL_FEATURES.copy(),
        "target": TARGET,
    }
