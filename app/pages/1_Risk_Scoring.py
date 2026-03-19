import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_data import load_policy_member_master

st.set_page_config(page_title="Risk Scoring", page_icon="📈", layout="wide")

@st.cache_data
def get_data():
    return load_policy_member_master()

df = get_data().copy()

st.title("Risk Scoring")

required = ["member_id", "policy_id"]
for col in required:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

risk_segment_col = "predicted_risk_segment" if "predicted_risk_segment" in df.columns else None
risk_score_col = "risk_score" if "risk_score" in df.columns else None

left, right = st.columns([1, 2])

with left:
    if risk_segment_col:
        segments = sorted(df[risk_segment_col].dropna().astype(str).unique().tolist())
        selected_segments = st.multiselect("Risk segment", segments, default=segments)
        df = df[df[risk_segment_col].astype(str).isin(selected_segments)]

with right:
    st.write("")

c1, c2, c3 = st.columns(3)
c1.metric("Policies / members", f"{len(df):,}")

if risk_segment_col:
    high_risk = df[risk_segment_col].astype(str).str.lower().isin(["high", "very_high"]).sum()
    c2.metric("High / Very high", f"{high_risk:,}")

if risk_score_col:
    avg_risk = pd.to_numeric(df[risk_score_col], errors="coerce").mean()
    c3.metric("Average risk score", f"{avg_risk:,.2f}" if pd.notna(avg_risk) else "N/A")

if risk_segment_col:
    st.subheader("Risk segment distribution")
    segment_counts = (
        df[risk_segment_col]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("segment")
        .reset_index(name="count")
    )
    st.bar_chart(segment_counts.set_index("segment"))

show_cols = [c for c in [
    "member_id",
    "policy_id",
    "predicted_risk_segment",
    "risk_score",
    "expected_annual_cost_proxy",
    "technical_premium_proxy",
    "commercial_premium_proxy",
] if c in df.columns]

st.subheader("Policy-member risk view")
st.dataframe(df[show_cols].head(200), use_container_width=True)
