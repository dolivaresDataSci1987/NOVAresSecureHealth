import pandas as pd
from src.config import (
    DASHBOARD_MASTER_POLICY_MEMBER,
    DASHBOARD_MASTER_PROVIDER,
    DASHBOARD_MASTER_PROSPECT,
    DASHBOARD_MASTER_DICTIONARY,
)


def load_policy_member_master() -> pd.DataFrame:
    return pd.read_csv(DASHBOARD_MASTER_POLICY_MEMBER)


def load_provider_master() -> pd.DataFrame:
    return pd.read_csv(DASHBOARD_MASTER_PROVIDER)


def load_prospect_master() -> pd.DataFrame:
    return pd.read_csv(DASHBOARD_MASTER_PROSPECT)


def load_dictionary() -> pd.DataFrame:
    return pd.read_csv(DASHBOARD_MASTER_DICTIONARY)
