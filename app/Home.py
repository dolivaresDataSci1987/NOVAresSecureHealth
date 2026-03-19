import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import APP_TITLE, APP_SUBTITLE
from src.data.load_data import (
    load_policy_member_master,
    load_provider_master,
    load_prospect_master,
)
from src.data.validation import dataset_overview

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
)

@st.cache_data
def get_policy_member():
    return load_policy_member_master()

@st.cache_data
def get_provider():
    return load_provider_master()

@st.cache_data
def get_prospect():
    return load_prospect_master()


# =========================
# SAFE LOADS
# =========================
try:
    df = get_policy_member()
except Exception as e:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.error(f"Could not load dashboard_master_policy_member.csv: {e}")
    st.info("Check that the file exists in data/dashboard/ and is correctly committed to GitHub.")
    st.stop()

try:
    provider_df = get_provider()
except Exception:
    provider_df = pd.DataFrame()

try:
    prospect_df = get_prospect()
except Exception:
    prospect_df = pd.DataFrame()


# =========================
# HEADER
# =========================
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

st.markdown(
    """
    **Disclaimer:** This dashboard is an analytical and decision-support demo.
    It does not replace actuarial, underwriting, fraud investigation, or clinical judgment.
    Some simulated pricing and recommendation fields may vary from real-world operational conditions.
    """
)

overview = dataset_overview(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows in master", f"{overview['rows']:,}")
c2.metric("Columns in master", f"{overview['columns']:,}")
c3.metric("Providers", f"{len(provider_df):,}")
c4.metric("Prospects", f"{len(prospect_df):,}")


# =========================
# MAIN KPIS
# =========================
st.subheader("Portfolio snapshot")

col_a, col_b, col_c = st.columns(3)

if "predicted_risk_segment" in df.columns:
    high_risk = df["predicted_risk_segment"].astype(str).str.lower().isin(["high", "very_high"]).sum()
    col_a.metric("High / Very high risk", f"{high_risk:,}")
else:
    col_a.metric("High / Very high risk", "N/A")

if "pricing_status" in df.columns:
    underpriced = df["pricing_status"].astype(str).str.lower().eq("underpriced").sum()
    col_b.metric("Underpriced policies", f"{underpriced:,}")
else:
    col_b.metric("Underpriced policies", "N/A")

if "member_abuse_severity" in df.columns:
    abuse = df["member_abuse_severity"].astype(str).str.lower().isin(["high", "very_high"]).sum()
    col_c.metric("High abuse severity", f"{abuse:,}")
else:
    col_c.metric("High abuse severity", "N/A")


# =========================
# OPTIONAL QUICK DISTRIBUTIONS
# =========================
dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    if "predicted_risk_segment" in df.columns:
        st.markdown("#### Risk segment distribution")
        risk_dist = (
            df["predicted_risk_segment"]
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("risk_segment")
            .reset_index(name="count")
        )
        st.bar_chart(risk_dist.set_index("risk_segment"))

with dist_col2:
    if "pricing_status" in df.columns:
        st.markdown("#### Pricing status distribution")
        pricing_dist = (
            df["pricing_status"]
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("pricing_status")
            .reset_index(name="count")
        )
        st.bar_chart(pricing_dist.set_index("pricing_status"))


# =========================
# DATA PREVIEW
# =========================
st.subheader("Sample of dashboard master")
st.dataframe(df.head(50), use_container_width=True)
