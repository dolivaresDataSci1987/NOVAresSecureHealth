import sys
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_data import load_policy_member_master

st.set_page_config(page_title="Case Explorer", page_icon="🔎", layout="wide")

@st.cache_data
def get_data():
    return load_policy_member_master()

df = get_data().copy()

st.title("Case Explorer")

member_options = sorted(df["member_id"].dropna().astype(str).unique().tolist()) if "member_id" in df.columns else []
policy_options = sorted(df["policy_id"].dropna().astype(str).unique().tolist()) if "policy_id" in df.columns else []

col1, col2 = st.columns(2)

selected_member = None
selected_policy = None

with col1:
    if member_options:
        selected_member = st.selectbox("Select member_id", [""] + member_options)

with col2:
    if policy_options:
        selected_policy = st.selectbox("Select policy_id", [""] + policy_options)

filtered = df.copy()

if selected_member:
    filtered = filtered[filtered["member_id"].astype(str) == selected_member]

if selected_policy:
    filtered = filtered[filtered["policy_id"].astype(str) == selected_policy]

st.subheader("Case result")

if filtered.empty:
    st.warning("No case found for the selected filters.")
else:
    row = filtered.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    if "member_id" in row.index:
        c1.metric("Member ID", str(row["member_id"]))
    if "policy_id" in row.index:
        c2.metric("Policy ID", str(row["policy_id"]))
    if "predicted_risk_segment" in row.index:
        c3.metric("Risk segment", str(row["predicted_risk_segment"]))
    if "pricing_status" in row.index:
        c4.metric("Pricing status", str(row["pricing_status"]))

    st.subheader("Full case record")
    case_df = pd.DataFrame(row).reset_index()
    case_df.columns = ["field", "value"]
    st.dataframe(case_df, use_container_width=True)
