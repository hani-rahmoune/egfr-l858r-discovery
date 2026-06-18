"""Convert raw IC50/Kd/Ki values to pIC50 and apply range/type filters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

UNIT_TO_MOLAR: dict[str, float] = {
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "nm": 1e-9,
    "pm": 1e-12,
    "fm": 1e-15,
    "mol/l": 1.0,
    "mmol/l": 1e-3,
    "umol/l": 1e-6,
    "nmol/l": 1e-9,
    "pmol/l": 1e-12,
}
SUPPORTED_TYPES = {"IC50", "Ki", "Kd", "EC50"}
PIC50_MIN = 3.0
PIC50_MAX = 12.0


def convert_to_molar(value: float, unit: str) -> float | None:
    factor = UNIT_TO_MOLAR.get(unit.strip().lower())
    return value * factor if factor is not None else None


def ic50_to_pic50(ic50_molar: float) -> float | None:
    if ic50_molar is None or ic50_molar <= 0 or np.isnan(ic50_molar):
        return None
    return -np.log10(ic50_molar)


def standardize_activity_column(
    df: pd.DataFrame,
    value_col: str = "activity_value",
    unit_col: str = "activity_units",
    type_col: str = "activity_type",
    pchembl_col: str = "pchembl_value",
) -> pd.DataFrame:
    df = df.copy()
    df["pic50"] = np.nan
    if pchembl_col in df.columns:
        pchembl = pd.to_numeric(df[pchembl_col], errors="coerce")
        mask = pchembl.notna()
        df.loc[mask, "pic50"] = pchembl[mask]
        logger.info(f"Used pChEMBL for {mask.sum()} records")
    # Fall back to manual unit conversion only for rows that lack a pChEMBL value
    needs = df["pic50"].isna()
    if needs.sum() > 0 and value_col in df.columns and unit_col in df.columns:

        def _convert(row):
            try:
                if pd.isna(row[value_col]) or pd.isna(row[unit_col]):
                    return None
                molar = convert_to_molar(float(row[value_col]), str(row[unit_col]))
                return ic50_to_pic50(molar) if molar is not None else None
            except Exception:
                return None

        converted = df[needs].apply(_convert, axis=1)
        df.loc[needs, "pic50"] = converted.values
        logger.info(f"Converted raw values for {converted.notna().sum()} records")
    return df


def filter_to_supported_types(
    df: pd.DataFrame, type_col: str = "activity_type"
) -> pd.DataFrame:
    if type_col not in df.columns:
        return df
    n = len(df)
    df = df[df[type_col].isin(SUPPORTED_TYPES)].copy()
    logger.info(f"filter_activity_types: {n} -> {len(df)}")
    return df.reset_index(drop=True)


def remove_invalid_activity(df: pd.DataFrame, pic50_col: str = "pic50") -> pd.DataFrame:
    n = len(df)
    df = df.dropna(subset=[pic50_col]).copy()
    pic50 = pd.to_numeric(df[pic50_col], errors="coerce")
    df = df[(pic50 >= PIC50_MIN) & (pic50 <= PIC50_MAX)].copy()
    df[pic50_col] = pd.to_numeric(df[pic50_col], errors="coerce")
    logger.info(f"remove_invalid_activity: {n} -> {len(df)}")
    return df.reset_index(drop=True)


def add_activity_label(
    df: pd.DataFrame,
    pic50_col: str = "pic50",
    active_threshold: float = 6.0,
    inactive_threshold: float = 5.0,
) -> pd.DataFrame:
    df = df.copy()
    df["binary_label"] = (df[pic50_col] >= active_threshold).astype(int)
    df["activity_class"] = pd.cut(
        df[pic50_col],
        bins=[-np.inf, inactive_threshold, active_threshold, np.inf],
        labels=["inactive", "gray", "active"],
    )
    return df


def run_activity_standardization(
    df: pd.DataFrame,
    value_col: str = "activity_value",
    unit_col: str = "activity_units",
    type_col: str = "activity_type",
    pchembl_col: str = "pchembl_value",
) -> pd.DataFrame:
    logger.info(f"Activity standardization start: {len(df)} records")
    df = filter_to_supported_types(df, type_col)
    df = standardize_activity_column(df, value_col, unit_col, type_col, pchembl_col)
    df = remove_invalid_activity(df)
    df = add_activity_label(df)
    logger.info(f"Activity standardization done: {len(df)} records")
    return df
