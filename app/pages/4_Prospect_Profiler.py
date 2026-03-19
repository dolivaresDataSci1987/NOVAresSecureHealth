import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_data import load_prospect_master

st.set_page_config(page_title="Prospect Profiler", page_icon="🧭", layout="wide")

@st.cache_data
def get_data():
    return load_prospect_master()

df = get_data().copy()

st.title("Prospect Profiler")

segment_col = "prospect_segment" if "prospect_segment" in df.columns else None
archetype_col = "archetype" if "archetype" in df.columns else None

left, right = st.columns(2)

with left:
    if segment_col:
        segments = sorted(df[segment_col].dropna().astype(str).unique().tolist())
        selected_segments = st.multiselect("Prospect segment", segments, default=segments)
        df = df[df[segment_col].astype(str).isin(selected_segments)]

with right:
    if archetype_col:
        archetypes = sorted(df[archetype_col].dropna().astype(str).unique().tolist())
        selected_archetypes = st.multiselect("Archetype", archetypes, default=archetypes)
        df = df[df[archetype_col].astype(str).isin(selected_archetypes)]

c1, c2, c3 = st.columns(3)
c1.metric("Prospects", f"{len(df):,}")

if "conversion_propensity" in df.columns:
    conv = pd.to_numeric(df["conversion_propensity"], errors="coerce").mean()
    c2.metric("Avg conversion propensity", f"{conv:,.2f}" if pd.notna(conv) else "N/A")

if "recommended_plan" in df.columns:
    top_plan = df["recommended_plan"].astype(str).mode()
    c3.metric("Top recommended plan", top_plan.iloc[0] if not top_plan.empty else "N/A")

if segment_col:
    st.subheader("Segment distribution")
    seg_counts = (
        df[segment_col]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("segment")
        .reset_index(name="count")
    )
    st.bar_chart(seg_counts.set_index("segment"))

cols = [c for c in [
    "prospect_id",
    "prospect_segment",
    "archetype",
    "conversion_propensity",
    "price_sensitivity",
    "recommended_plan",
    "recommended_plan_2",
] if c in df.columns]

st.subheader("Prospect recommendations")
st.dataframe(df[cols].head(200), use_container_width=True)
