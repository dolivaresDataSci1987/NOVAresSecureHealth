import pandas as pd

def dataset_shape(df: pd.DataFrame) -> tuple:
    return df.shape

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.isna().sum().reset_index()
    out.columns = ["column", "missing_n"]
    out["missing_pct"] = out["missing_n"] / len(df)
    return out.sort_values("missing_pct", ascending=False)

def duplicated_rows(df: pd.DataFrame) -> int:
    return df.duplicated().sum()

def unique_count(df: pd.DataFrame, col: str) -> int:
    return df[col].nunique(dropna=True)
