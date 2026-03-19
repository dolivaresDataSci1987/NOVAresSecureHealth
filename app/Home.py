import sys
from pathlib import Path

import streamlit as st
import pandas as pd

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

df = get_policy_member()
provider_df = get_provider()
prospect_df = get_prospect()

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

st.subheader("Portfolio snapshot")

col_a, col_b, col_c = st.columns(3)

if "predicted_risk_segment" in df.columns:
    high_risk = df["predicted_risk_segment"].astype(str).str.lower().isin(["high", "very_high"]).sum()
    col_a.metric("High / Very high risk", f"{high_risk:,}")

if "pricing_status" in df.columns:
    underpriced = df["pricing_status"].astype(str).str.lower().eq("underpriced").sum()
    col_b.metric("Underpriced policies", f"{underpriced:,}")

if "member_abuse_severity" in df.columns:
    abuse = df["member_abuse_severity"].astype(str).str.lower().isin(["high", "very_high"]).sum()
    col_c.metric("High abuse severity", f"{abuse:,}")

st.subheader("Sample of dashboard master")
st.dataframe(df.head(50), use_container_width=True)
