import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_data import load_policy_member_master, load_provider_master

st.set_page_config(page_title="Fraud / Abuse", page_icon="🚨", layout="wide")

@st.cache_data
def get_member_data():
    return load_policy_member_master()

@st.cache_data
def get_provider_data():
    return load_provider_master()

df = get_member_data().copy()
provider_df = get_provider_data().copy()

st.title("Fraud / Abuse")

tab1, tab2 = st.tabs(["Member abuse", "Provider fraud"])

with tab1:
    severity_col = "member_abuse_severity" if "member_abuse_severity" in df.columns else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")

    if severity_col:
        high_cases = df[severity_col].astype(str).str.lower().isin(["high", "very_high"]).sum()
        c2.metric("High / Very high abuse", f"{high_cases:,}")

    if "member_abuse_flag" in df.columns:
        flagged = df["member_abuse_flag"].fillna(0).astype(int).sum()
        c3.metric("Flagged members", f"{flagged:,}")

    if severity_col:
        sev_counts = (
            df[severity_col]
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("severity")
            .reset_index(name="count")
        )
        st.bar_chart(sev_counts.set_index("severity"))

    cols = [c for c in [
        "member_id",
        "policy_id",
        "member_abuse_flag",
        "member_abuse_score",
        "member_abuse_severity",
        "claims_count",
        "observed_cost_sum",
    ] if c in df.columns]

    st.dataframe(df[cols].head(200), use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    c1.metric("Providers", f"{len(provider_df):,}")

    if "provider_fraud_flag" in provider_df.columns:
        flagged = provider_df["provider_fraud_flag"].fillna(0).astype(int).sum()
        c2.metric("Flagged providers", f"{flagged:,}")

    if "provider_fraud_severity" in provider_df.columns:
        sev_counts = (
            provider_df["provider_fraud_severity"]
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("severity")
            .reset_index(name="count")
        )
        st.bar_chart(sev_counts.set_index("severity"))

    cols = [c for c in [
        "provider_id",
        "provider_name",
        "provider_type",
        "provider_fraud_flag",
        "provider_fraud_score",
        "provider_fraud_severity",
    ] if c in provider_df.columns]

    st.dataframe(provider_df[cols].head(200), use_container_width=True)
