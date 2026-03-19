from pathlib import Path
import pandas as pd

from src.config import (
    DASHBOARD_MASTER_POLICY_MEMBER,
    DASHBOARD_MASTER_PROVIDER,
    DASHBOARD_MASTER_PROSPECT,
    DASHBOARD_MASTER_DICTIONARY,
)


def _read_csv_safe(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_name}. Expected path: {path}"
        )
    return pd.read_csv(path)


def load_policy_member_master() -> pd.DataFrame:
    return _read_csv_safe(
        DASHBOARD_MASTER_POLICY_MEMBER,
        "dashboard_master_policy_member.csv"
    )


def load_provider_master() -> pd.DataFrame:
    return _read_csv_safe(
        DASHBOARD_MASTER_PROVIDER,
        "dashboard_master_provider.csv"
    )


def load_prospect_master() -> pd.DataFrame:
    return _read_csv_safe(
        DASHBOARD_MASTER_PROSPECT,
        "dashboard_master_prospect.csv"
    )


def load_dictionary() -> pd.DataFrame:
    return _read_csv_safe(
        DASHBOARD_MASTER_DICTIONARY,
        "dashboard_master_dictionary.csv"
    )
