from pathlib import Path
import pandas as pd

from src.config import RAW_DIR


def load_insured_members() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "insured_members.csv")


def load_policies() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "policies.csv")


def load_providers() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "providers.csv")


def load_claims() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "claims_corrected.csv")


def load_member_year_features() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "member_year_features_corrected.csv")


def load_provider_month_features() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "provider_month_features.csv")


def load_prospect_survey() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "prospect_survey_synthetic.csv")
