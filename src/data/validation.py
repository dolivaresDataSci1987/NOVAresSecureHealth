import pandas as pd


def check_required_columns(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    missing = [col for col in required_columns if col not in df.columns]
    return missing


def dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
    }


def safe_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")
