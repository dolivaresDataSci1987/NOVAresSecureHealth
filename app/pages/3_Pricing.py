import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_data import load_policy_member_master

st.set_page_config(page_title="Pricing", page_icon="💰", layout="wide")

@st.cache_data
def get_data():
    return load_policy_member_master()

df = get_data().copy()

st.title("Pricing")

pricing_col = "pricing_status" if "pricing_status" in df.columns else None

if pricing_col:
    statuses = sorted(df[pricing_col].dropna().astype(str).unique().tolist())
    selected = st.multiselect("Pricing status", statuses, default=statuses)
    df = df[df[pricing_col].astype(str).isin(selected)]

c1, c2, c3 = st.columns(3)
c1.metric("Policies", f"{len(df):,}")

if "commercial_premium_proxy" in df.columns:
    current_avg = pd.to_numeric(df["commercial_premium_proxy"], errors="coerce").mean()
    c2.metric("Avg commercial premium", f"{current_avg:,.2f}" if pd.notna(current_avg) else "N/A")

if "technical_premium_proxy" in df.columns:
    tech_avg = pd.to_numeric(df["technical_premium_proxy"], errors="coerce").mean()
    c3.metric("Avg technical premium", f"{tech_avg:,.2f}" if pd.notna(tech_avg) else "N/A")

if pricing_col:
    st.subheader("Pricing status distribution")
    counts = (
        df[pricing_col]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("pricing_status")
        .reset_index(name="count")
    )
    st.bar_chart(counts.set_index("pricing_status"))

cols = [c for c in [
    "member_id",
    "policy_id",
    "pricing_status",
    "observed_cost_sum",
    "expected_annual_cost_proxy",
    "technical_premium_proxy",
    "commercial_premium_proxy",
    "suggested_premium_low",
    "suggested_premium_mid",
    "suggested_premium_high",
] if c in df.columns]

st.subheader("Pricing review table")
st.dataframe(df[cols].head(200), use_container_width=True)
