import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_data import load_policy_member_master


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Risk Scoring",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# DATA
# =========================================================
@st.cache_data
def get_data():
    return load_policy_member_master()


df = get_data().copy()

st.title("📈 Risk Scoring")
st.caption(
    "Vista analítica del riesgo técnico de la cartera: distribución, concentración, "
    "coste esperado y señales de revisión."
)

required = ["member_id", "policy_id"]
for col in required:
    if col not in df.columns:
        st.error(f"Falta la columna requerida: {col}")
        st.stop()

risk_segment_col = "predicted_risk_segment" if "predicted_risk_segment" in df.columns else None
risk_score_col = "risk_score" if "risk_score" in df.columns else None
expected_cost_col = "expected_annual_cost_proxy" if "expected_annual_cost_proxy" in df.columns else None
tech_premium_col = "technical_premium_proxy" if "technical_premium_proxy" in df.columns else None
comm_premium_col = "commercial_premium_proxy" if "commercial_premium_proxy" in df.columns else None


# =========================================================
# HELPERS
# =========================================================
def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def fmt_int(x):
    if pd.isna(x):
        return "N/A"
    return f"{int(round(x)):,}".replace(",", ".")


def fmt_num(x, decimals=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(x, decimals=1):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{decimals}f}%"


def safe_mean(df_, col):
    if col and col in df_.columns:
        return to_numeric(df_[col]).mean()
    return np.nan


def safe_sum(df_, col):
    if col and col in df_.columns:
        return to_numeric(df_[col]).sum()
    return np.nan


def normalize_segment_name(x):
    if pd.isna(x):
        return "Sin dato"
    txt = str(x).strip()
    lower = txt.lower()

    mapping = {
        "very_low": "Muy bajo",
        "low": "Bajo",
        "medium": "Medio",
        "moderate": "Medio",
        "high": "Alto",
        "very_high": "Muy alto",
        "muy bajo": "Muy bajo",
        "bajo": "Bajo",
        "medio": "Medio",
        "alto": "Alto",
        "muy alto": "Muy alto",
    }
    return mapping.get(lower, txt)


def segment_order_key(x):
    order = {
        "Muy bajo": 1,
        "Bajo": 2,
        "Medio": 3,
        "Alto": 4,
        "Muy alto": 5,
        "Sin dato": 99,
    }
    return order.get(x, 50)


def build_score_buckets(series):
    s = to_numeric(series).dropna()
    if s.empty:
        return None

    min_v = float(np.floor(s.min()))
    max_v = float(np.ceil(s.max()))

    if min_v == max_v:
        bins = [min_v - 1, min_v, min_v + 1]
    else:
        bins = np.linspace(min_v, max_v, 11)

    bucketed = pd.cut(s, bins=bins, include_lowest=True, duplicates="drop")
    counts = bucketed.value_counts().sort_index()

    out = counts.reset_index()
    out.columns = ["tramo", "casos"]
    out["tramo"] = out["tramo"].astype(str)
    return out


def top_n_cases(df_, n=15):
    df_top = df_.copy()

    if risk_score_col and risk_score_col in df_top.columns:
        df_top["_risk_score_num"] = to_numeric(df_top[risk_score_col])
    else:
        df_top["_risk_score_num"] = np.nan

    if expected_cost_col and expected_cost_col in df_top.columns:
        df_top["_expected_cost_num"] = to_numeric(df_top[expected_cost_col])
    else:
        df_top["_expected_cost_num"] = np.nan

    sort_cols = [c for c in ["_risk_score_num", "_expected_cost_num"] if c in df_top.columns]
    if sort_cols:
        df_top = df_top.sort_values(sort_cols, ascending=False)

    cols = [
        "member_id",
        "policy_id",
        risk_segment_col,
        risk_score_col,
        expected_cost_col,
        tech_premium_col,
        comm_premium_col,
    ]
    cols = [c for c in cols if c and c in df_top.columns]

    return df_top[cols].head(n)


# =========================================================
# NORMALIZACION BASICA
# =========================================================
if risk_segment_col:
    df[risk_segment_col] = df[risk_segment_col].apply(normalize_segment_name)

if risk_score_col:
    df[risk_score_col] = to_numeric(df[risk_score_col])

if expected_cost_col:
    df[expected_cost_col] = to_numeric(df[expected_cost_col])

if tech_premium_col:
    df[tech_premium_col] = to_numeric(df[tech_premium_col])

if comm_premium_col:
    df[comm_premium_col] = to_numeric(df[comm_premium_col])


# =========================================================
# SIDEBAR / FILTROS
# =========================================================
st.sidebar.header("Filtros")

df_filtered = df.copy()

if risk_segment_col:
    segment_options = sorted(
        df_filtered[risk_segment_col].dropna().astype(str).unique().tolist(),
        key=segment_order_key
    )
    selected_segments = st.sidebar.multiselect(
        "Segmento de riesgo",
        options=segment_options,
        default=segment_options
    )
    if selected_segments:
        df_filtered = df_filtered[df_filtered[risk_segment_col].astype(str).isin(selected_segments)]

if risk_score_col and df_filtered[risk_score_col].notna().any():
    score_min = float(df_filtered[risk_score_col].min())
    score_max = float(df_filtered[risk_score_col].max())

    if score_min < score_max:
        selected_range = st.sidebar.slider(
            "Rango de risk score",
            min_value=float(score_min),
            max_value=float(score_max),
            value=(float(score_min), float(score_max))
        )
        df_filtered = df_filtered[
            df_filtered[risk_score_col].between(selected_range[0], selected_range[1], inclusive="both")
        ]

search_policy = st.sidebar.text_input("Buscar policy_id")
if search_policy:
    df_filtered = df_filtered[
        df_filtered["policy_id"].astype(str).str.contains(search_policy, case=False, na=False)
    ]

search_member = st.sidebar.text_input("Buscar member_id")
if search_member:
    df_filtered = df_filtered[
        df_filtered["member_id"].astype(str).str.contains(search_member, case=False, na=False)
    ]


# =========================================================
# KPIs
# =========================================================
total_records = len(df_filtered)
high_risk_count = np.nan
high_risk_pct = np.nan

if risk_segment_col:
    high_mask = df_filtered[risk_segment_col].astype(str).str.lower().isin(["alto", "muy alto", "high", "very_high"])
    high_risk_count = int(high_mask.sum())
    high_risk_pct = (high_risk_count / total_records * 100) if total_records > 0 else np.nan

avg_risk_score = safe_mean(df_filtered, risk_score_col)
total_expected_cost = safe_sum(df_filtered, expected_cost_col)
avg_expected_cost = safe_mean(df_filtered, expected_cost_col)
avg_tech_premium = safe_mean(df_filtered, tech_premium_col)
avg_comm_premium = safe_mean(df_filtered, comm_premium_col)

pricing_gap = np.nan
if pd.notna(avg_comm_premium) and pd.notna(avg_expected_cost):
    pricing_gap = avg_comm_premium - avg_expected_cost

st.markdown("## Resumen ejecutivo")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Registros analizados", fmt_int(total_records))
k2.metric("Riesgo alto / muy alto", fmt_int(high_risk_count) if pd.notna(high_risk_count) else "N/A",
          delta=fmt_pct(high_risk_pct) if pd.notna(high_risk_pct) else None)
k3.metric("Risk score medio", fmt_num(avg_risk_score))
k4.metric("Coste esperado medio", fmt_num(avg_expected_cost))
k5.metric("Prima comercial media", fmt_num(avg_comm_premium))


# =========================================================
# BLOQUE DE LECTURA DE NEGOCIO
# =========================================================
insights = []

if total_records > 0:
    insights.append(f"La vista actual cubre **{fmt_int(total_records)} registros** de póliza-miembro.")

if pd.notna(high_risk_pct):
    insights.append(
        f"El **{fmt_pct(high_risk_pct)}** de la cartera filtrada está en segmentos **alto o muy alto**, "
        "lo que ayuda a identificar la parte más tensionada del portfolio."
    )

if pd.notna(avg_expected_cost) and pd.notna(avg_comm_premium):
    if pricing_gap >= 0:
        insights.append(
            f"La **prima comercial media** se sitúa **{fmt_num(pricing_gap)}** por encima del "
            f"**coste esperado medio**, señal de mayor holgura comercial en la vista filtrada."
        )
    else:
        insights.append(
            f"La **prima comercial media** queda **{fmt_num(abs(pricing_gap))}** por debajo del "
            f"**coste esperado medio**, posible señal de presión técnica o infrapricing."
        )

if pd.notna(total_expected_cost):
    insights.append(
        f"El **coste esperado agregado** de los registros visibles asciende a **{fmt_num(total_expected_cost)}**."
    )

with st.container(border=True):
    st.markdown("### Lectura de negocio")
    if insights:
        for item in insights:
            st.markdown(f"- {item}")
    else:
        st.info("No hay suficiente información para construir una lectura ejecutiva en la vista actual.")


# =========================================================
# DISTRIBUCION + CONCENTRACION
# =========================================================
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### Distribución por segmento de riesgo")
    if risk_segment_col:
        segment_dist = (
            df_filtered[risk_segment_col]
            .fillna("Sin dato")
            .value_counts()
            .rename_axis("segmento")
            .reset_index(name="casos")
        )
        segment_dist["orden"] = segment_dist["segmento"].map(segment_order_key)
        segment_dist = segment_dist.sort_values("orden").drop(columns="orden")
        st.bar_chart(segment_dist.set_index("segmento"))
        st.dataframe(segment_dist, use_container_width=True, hide_index=True)
    else:
        st.info("No existe la columna de segmento de riesgo en el dataset.")

with col_b:
    st.markdown("### Distribución del risk score")
    if risk_score_col and df_filtered[risk_score_col].notna().any():
        buckets = build_score_buckets(df_filtered[risk_score_col])
        if buckets is not None and not buckets.empty:
            st.bar_chart(buckets.set_index("tramo"))
            st.dataframe(buckets, use_container_width=True, hide_index=True)
        else:
            st.info("No se pudo construir la distribución del risk score.")
    else:
        st.info("No existe información suficiente de risk score para mostrar tramos.")


# =========================================================
# TABLA POR SEGMENTO
# =========================================================
st.markdown("### Comparativa por segmento")

if risk_segment_col:
    agg_dict = {
        "member_id": "count",
        "policy_id": pd.Series.nunique
    }

    if risk_score_col:
        agg_dict[risk_score_col] = "mean"
    if expected_cost_col:
        agg_dict[expected_cost_col] = "mean"
    if tech_premium_col:
        agg_dict[tech_premium_col] = "mean"
    if comm_premium_col:
        agg_dict[comm_premium_col] = "mean"

    segment_summary = (
        df_filtered
        .groupby(risk_segment_col, dropna=False)
        .agg(agg_dict)
        .reset_index()
        .rename(columns={
            risk_segment_col: "segmento_riesgo",
            "member_id": "registros",
            "policy_id": "polizas_unicas",
            risk_score_col: "risk_score_medio" if risk_score_col else "risk_score_medio",
            expected_cost_col: "coste_esperado_medio" if expected_cost_col else "coste_esperado_medio",
            tech_premium_col: "prima_tecnica_media" if tech_premium_col else "prima_tecnica_media",
            comm_premium_col: "prima_comercial_media" if comm_premium_col else "prima_comercial_media",
        })
    )

    if "prima_comercial_media" in segment_summary.columns and "coste_esperado_medio" in segment_summary.columns:
        segment_summary["gap_pricing"] = (
            segment_summary["prima_comercial_media"] - segment_summary["coste_esperado_medio"]
        )

    segment_summary["orden"] = segment_summary["segmento_riesgo"].map(segment_order_key)
    segment_summary = segment_summary.sort_values("orden").drop(columns="orden")

    format_dict = {}
    for c in [
        "risk_score_medio",
        "coste_esperado_medio",
        "prima_tecnica_media",
        "prima_comercial_media",
        "gap_pricing",
    ]:
        if c in segment_summary.columns:
            format_dict[c] = lambda x: fmt_num(x)

    st.dataframe(
        segment_summary.style.format(format_dict),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No se puede construir la comparativa por segmento porque falta la columna de segmentación.")


# =========================================================
# CASOS PRIORITARIOS
# =========================================================
st.markdown("### Casos prioritarios para revisión")

priority_df = top_n_cases(df_filtered, n=20)

if not priority_df.empty:
    st.caption(
        "Se muestran los registros con mayor score y/o mayor coste esperado para facilitar revisión técnica o comercial."
    )
    st.dataframe(priority_df, use_container_width=True, hide_index=True)
else:
    st.info("No hay casos suficientes para construir una lista prioritaria.")


# =========================================================
# TABLA GENERAL
# =========================================================
st.markdown("### Vista detallada de póliza-miembro")

show_cols = [
    "member_id",
    "policy_id",
    risk_segment_col,
    risk_score_col,
    expected_cost_col,
    tech_premium_col,
    comm_premium_col,
]
show_cols = [c for c in show_cols if c and c in df_filtered.columns]

if show_cols:
    st.dataframe(
        df_filtered[show_cols],
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("No hay columnas disponibles para mostrar la vista detallada.")
