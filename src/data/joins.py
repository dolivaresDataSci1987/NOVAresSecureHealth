import pandas as pd


def build_case_view(
    policy_member_df: pd.DataFrame,
    provider_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    df = policy_member_df.copy()

    if provider_df is not None and "provider_id" in df.columns and "provider_id" in provider_df.columns:
        provider_cols = [c for c in provider_df.columns if c != "provider_id"]
        provider_subset = provider_df[["provider_id"] + provider_cols].drop_duplicates("provider_id")
        df = df.merge(provider_subset, on="provider_id", how="left", suffixes=("", "_provider"))

    return df
