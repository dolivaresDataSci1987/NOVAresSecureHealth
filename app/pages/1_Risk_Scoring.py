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
    page_title="Riesgo de pérdida técnica",
    page_icon="📉",
    layout="wide"
)


# =========================================================
# DATA
# =========================================================
@st.cache_data
def get_data():
    return load_policy_member_master()


df = get_data().copy()

st.title("📉 Riesgo de pérdida técnica")
st.caption(
    "Análisis de probabilidad de pérdida para la aseguradora: "
    "cuándo el coste aprobado supera la prima anual y qué segmentos, pólizas o perfiles tensionan la rentabilidad."
)

required = ["member_id", "policy_id", "premium_annual", "approved_cost_sum"]
for col in required:
    if col not in df.columns:
        st.error(f"Falta la columna requerida: {col}")
        st.stop()


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


def build_margin_business_buckets(series):
    s = pd.to_numeric(series, errors="coerce")

    bins = [-np.inf, -50000, -10000, -1000, 1000, 10000, 50000, np.inf]
    labels = [
        "Pérdida severa (< -50k)",
        "Pérdida alta (-50k a -10k)",
        "Pérdida moderada (-10k a -1k)",
        "Equilibrio técnico (-1k a 1k)",
        "Margen positivo bajo (1k a 10k)",
        "Margen positivo medio (10k a 50k)",
        "Margen positivo alto (> 50k)",
    ]

    bucketed = pd.cut(s, bins=bins, labels=labels, include_lowest=True)
    out = (
        bucketed.value_counts(dropna=False)
        .reindex(labels, fill_value=0)
        .reset_index()
    )
    out.columns = ["tramo", "casos"]
    return out


def build_ratio_business_buckets(series):
    s = pd.to_numeric(series, errors="coerce")

    bins = [-np.inf, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, np.inf]
    labels = [
        "Muy rentable (<= 0,5)",
        "Rentable (0,5–0,8)",
        "Ajustado (0,8–1,0)",
        "Pérdida leve (1,0–1,2)",
        "Pérdida moderada (1,2–1,5)",
        "Pérdida alta (1,5–2,0)",
        "Pérdida severa (> 2,0)",
    ]

    bucketed = pd.cut(s, bins=bins, labels=labels, include_lowest=True)
    out = (
        bucketed.value_counts(dropna=False)
        .reindex(labels, fill_value=0)
        .reset_index()
    )
    out.columns = ["tramo", "casos"]
    return out


# =========================================================
# COLUMNAS DISPONIBLES
# =========================================================
risk_segment_col = "predicted_risk_segment" if "predicted_risk_segment" in df.columns else None
risk_prob_col = "predicted_risk_probability" if "predicted_risk_probability" in df.columns else None
plan_type_col = "plan_type" if "plan_type" in df.columns else None
plan_tier_col = "plan_tier" if "plan_tier" in df.columns else None
coverage_scope_col = "coverage_scope" if "coverage_scope" in df.columns else None
network_col = "provider_network_type" if "provider_network_type" in df.columns else None
claims_col = "claims_count" if "claims_count" in df.columns else ("claims_n" if "claims_n" in df.columns else None)
chronic_flag_col = "chronic_condition_flag" if "chronic_condition_flag" in df.columns else None
chronic_count_col = "chronic_condition_count" if "chronic_condition_count" in df.columns else None
abuse_score_col = "member_abuse_score" if "member_abuse_score" in df.columns else None
flagged_cost_col = "flagged_provider_cost_sum" if "flagged_provider_cost_sum" in df.columns else None
pricing_adequacy_col = "pricing_adequacy_ratio" if "pricing_adequacy_ratio" in df.columns else None


# =========================================================
# NORMALIZACIÓN Y MÉTRICAS DERIVADAS
# =========================================================
df["premium_annual"] = to_numeric(df["premium_annual"])
df["approved_cost_sum"] = to_numeric(df["approved_cost_sum"])

if risk_prob_col:
    df[risk_prob_col] = to_numeric(df[risk_prob_col])

if claims_col:
    df[claims_col] = to_numeric(df[claims_col])

if chronic_count_col:
    df[chronic_count_col] = to_numeric(df[chronic_count_col])

if abuse_score_col:
    df[abuse_score_col] = to_numeric(df[abuse_score_col])

if flagged_cost_col:
    df[flagged_cost_col] = to_numeric(df[flagged_cost_col])

if pricing_adequacy_col:
    df[pricing_adequacy_col] = to_numeric(df[pricing_adequacy_col])

if risk_segment_col:
    df[risk_segment_col] = df[risk_segment_col].apply(normalize_segment_name)

df["margen_tecnico"] = df["premium_annual"] - df["approved_cost_sum"]
df["perdida_tecnica_flag"] = df["approved_cost_sum"] > df["premium_annual"]
df["ratio_coste_prima"] = np.where(
    df["premium_annual"] > 0,
    df["approved_cost_sum"] / df["premium_annual"],
    np.nan
)

df["severidad_perdida"] = np.where(
    df["margen_tecnico"] < 0,
    -df["margen_tecnico"],
    0
)

if "premium_monthly" in df.columns:
    df["premium_monthly"] = to_numeric(df["premium_monthly"])


# =========================================================
# SIDEBAR
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
        df_filtered = df_filtered[df_filtered[risk_segment_col].isin(selected_segments)]

if plan_type_col:
    plan_options = sorted(df_filtered[plan_type_col].dropna().astype(str).unique().tolist())
    selected_plans = st.sidebar.multiselect(
        "Tipo de plan",
        options=plan_options,
        default=plan_options
    )
    if selected_plans:
        df_filtered = df_filtered[df_filtered[plan_type_col].astype(str).isin(selected_plans)]

if coverage_scope_col:
    coverage_options = sorted(df_filtered[coverage_scope_col].dropna().astype(str).unique().tolist())
    selected_coverage = st.sidebar.multiselect(
        "Cobertura",
        options=coverage_options,
        default=coverage_options
    )
    if selected_coverage:
        df_filtered = df_filtered[df_filtered[coverage_scope_col].astype(str).isin(selected_coverage)]

if network_col:
    network_options = sorted(df_filtered[network_col].dropna().astype(str).unique().tolist())
    selected_network = st.sidebar.multiselect(
        "Red de proveedores",
        options=network_options,
        default=network_options
    )
    if selected_network:
        df_filtered = df_filtered[df_filtered[network_col].astype(str).isin(selected_network)]

solo_perdidas = st.sidebar.checkbox("Mostrar solo casos con pérdida técnica", value=False)
if solo_perdidas:
    df_filtered = df_filtered[df_filtered["perdida_tecnica_flag"]]

if risk_prob_col and df_filtered[risk_prob_col].notna().any():
    prob_min = float(df_filtered[risk_prob_col].min())
    prob_max = float(df_filtered[risk_prob_col].max())

    if prob_min < prob_max:
        prob_range = st.sidebar.slider(
            "Rango de probabilidad de riesgo",
            min_value=float(prob_min),
            max_value=float(prob_max),
            value=(float(prob_min), float(prob_max))
        )
        df_filtered = df_filtered[
            df_filtered[risk_prob_col].between(prob_range[0], prob_range[1], inclusive="both")
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
# KPI BASE
# =========================================================
n_registros = len(df_filtered)
prima_total = df_filtered["premium_annual"].sum()
coste_total = df_filtered["approved_cost_sum"].sum()
margen_total = df_filtered["margen_tecnico"].sum()
ratio_global = (coste_total / prima_total) if prima_total > 0 else np.nan
pct_perdida = df_filtered["perdida_tecnica_flag"].mean() * 100 if n_registros > 0 else np.nan
n_perdida = int(df_filtered["perdida_tecnica_flag"].sum()) if n_registros > 0 else 0
prima_media = df_filtered["premium_annual"].mean()
coste_medio = df_filtered["approved_cost_sum"].mean()
margen_medio = df_filtered["margen_tecnico"].mean()

if risk_prob_col:
    riesgo_prob_medio = df_filtered[risk_prob_col].mean()
else:
    riesgo_prob_medio = np.nan


# =========================================================
# CABECERA EJECUTIVA
# =========================================================
st.markdown("## Resumen técnico de rentabilidad")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Registros analizados", fmt_int(n_registros))
k2.metric("Casos con pérdida", fmt_int(n_perdida), delta=fmt_pct(pct_perdida) if pd.notna(pct_perdida) else None)
k3.metric("Prima total anual", fmt_num(prima_total))
k4.metric("Coste aprobado total", fmt_num(coste_total))
k5.metric("Margen técnico total", fmt_num(margen_total))
k6.metric("Ratio coste / prima", fmt_num(ratio_global))


# =========================================================
# LECTURA DE NEGOCIO
# =========================================================
insights = []

if n_registros > 0:
    insights.append(
        f"La vista actual incluye **{fmt_int(n_registros)} registros** de póliza-miembro."
    )

if pd.notna(pct_perdida):
    insights.append(
        f"El **{fmt_pct(pct_perdida)}** de los registros está en **pérdida técnica**, "
        f"es decir, con **coste aprobado superior a la prima anual**."
    )

if pd.notna(margen_total):
    if margen_total < 0:
        insights.append(
            f"El **margen técnico agregado** es de **{fmt_num(margen_total)}**, "
            "lo que indica presión negativa sobre la rentabilidad de la cartera filtrada."
        )
    else:
        insights.append(
            f"El **margen técnico agregado** es de **{fmt_num(margen_total)}**, "
            "lo que sugiere una cartera filtrada todavía rentable."
        )

if pd.notna(ratio_global):
    if ratio_global > 1:
        insights.append(
            f"El **ratio coste/prima medio** es **{fmt_num(ratio_global)}**, "
            "por encima de 1, lo que implica que globalmente el coste está superando el ingreso por prima."
        )
    else:
        insights.append(
            f"El **ratio coste/prima medio** es **{fmt_num(ratio_global)}**, "
            "lo que sugiere equilibrio o margen positivo en el conjunto filtrado."
        )

if risk_prob_col and pd.notna(riesgo_prob_medio):
    insights.append(
        f"La **probabilidad media de riesgo** estimada en esta vista es **{fmt_num(riesgo_prob_medio)}**."
    )

with st.container(border=True):
    st.markdown("### Lectura ejecutiva")
    for item in insights:
        st.markdown(f"- {item}")


# =========================================================
# DISTRIBUCIÓN DEL PROBLEMA
# =========================================================
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Distribución del margen técnico")
    st.caption("Tramos de negocio para entender rápidamente dónde se concentra la pérdida o el margen.")
    margen_dist = build_margin_business_buckets(df_filtered["margen_tecnico"])
    if not margen_dist.empty:
        st.bar_chart(margen_dist.set_index("tramo"))
        st.dataframe(margen_dist, use_container_width=True, hide_index=True)
    else:
        st.info("No hay suficiente información para mostrar la distribución del margen.")

with c2:
    st.markdown("### Distribución del ratio coste / prima")
    st.caption("Valores por encima de 1 implican que el coste supera la prima y la aseguradora entra en pérdida.")
    ratio_dist = build_ratio_business_buckets(df_filtered["ratio_coste_prima"])
    if not ratio_dist.empty:
        st.bar_chart(ratio_dist.set_index("tramo"))
        st.dataframe(ratio_dist, use_container_width=True, hide_index=True)
    else:
        st.info("No hay suficiente información para mostrar la distribución del ratio coste/prima.")


# =========================================================
# DÓNDE SE PIERDE DINERO
# =========================================================
st.markdown("## Dónde se concentra la pérdida")

g1, g2 = st.columns(2)

with g1:
    st.markdown("### % de casos con pérdida por segmento de riesgo")
    if risk_segment_col:
        seg_loss = (
            df_filtered
            .groupby(risk_segment_col, dropna=False)
            .agg(
                registros=("member_id", "count"),
                pct_perdida=("perdida_tecnica_flag", "mean")
            )
            .reset_index()
        )
        seg_loss["pct_perdida"] = seg_loss["pct_perdida"] * 100
        seg_loss["orden"] = seg_loss[risk_segment_col].map(segment_order_key)
        seg_loss = seg_loss.sort_values("orden").drop(columns="orden")
        st.bar_chart(seg_loss.set_index(risk_segment_col)[["pct_perdida"]])
        st.dataframe(
            seg_loss.rename(columns={
                risk_segment_col: "segmento_riesgo",
                "pct_perdida": "%_perdida"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No existe columna de segmento de riesgo.")

with g2:
    st.markdown("### % de casos con pérdida por tipo de plan")
    if plan_type_col:
        plan_loss = (
            df_filtered
            .groupby(plan_type_col, dropna=False)
            .agg(
                registros=("member_id", "count"),
                pct_perdida=("perdida_tecnica_flag", "mean")
            )
            .reset_index()
        )
        plan_loss["pct_perdida"] = plan_loss["pct_perdida"] * 100
        plan_loss = plan_loss.sort_values("pct_perdida", ascending=False)
        st.bar_chart(plan_loss.set_index(plan_type_col)[["pct_perdida"]])
        st.dataframe(
            plan_loss.rename(columns={
                plan_type_col: "tipo_plan",
                "pct_perdida": "%_perdida"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No existe columna de tipo de plan.")


# =========================================================
# COSTE VS PRIMA POR GRUPOS
# =========================================================
st.markdown("## Coste medio vs prima media por grupo")

h1, h2 = st.columns(2)

with h1:
    st.markdown("### Comparativa por segmento de riesgo")
    if risk_segment_col:
        seg_compare = (
            df_filtered
            .groupby(risk_segment_col, dropna=False)
            .agg(
                registros=("member_id", "count"),
                prima_media=("premium_annual", "mean"),
                coste_medio=("approved_cost_sum", "mean"),
                margen_medio=("margen_tecnico", "mean"),
                pct_perdida=("perdida_tecnica_flag", "mean")
            )
            .reset_index()
        )
        seg_compare["pct_perdida"] = seg_compare["pct_perdida"] * 100
        seg_compare["orden"] = seg_compare[risk_segment_col].map(segment_order_key)
        seg_compare = seg_compare.sort_values("orden").drop(columns="orden")

        chart_seg = seg_compare[[risk_segment_col, "prima_media", "coste_medio"]].set_index(risk_segment_col)
        st.bar_chart(chart_seg)
        st.dataframe(
            seg_compare.rename(columns={
                risk_segment_col: "segmento_riesgo",
                "pct_perdida": "%_perdida"
            }).style.format({
                "prima_media": lambda x: fmt_num(x),
                "coste_medio": lambda x: fmt_num(x),
                "margen_medio": lambda x: fmt_num(x),
                "%_perdida": lambda x: fmt_pct(x),
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No existe columna de segmento de riesgo.")

with h2:
    st.markdown("### Comparativa por tipo de plan")
    if plan_type_col:
        plan_compare = (
            df_filtered
            .groupby(plan_type_col, dropna=False)
            .agg(
                registros=("member_id", "count"),
                prima_media=("premium_annual", "mean"),
                coste_medio=("approved_cost_sum", "mean"),
                margen_medio=("margen_tecnico", "mean"),
                pct_perdida=("perdida_tecnica_flag", "mean")
            )
            .reset_index()
        )
        plan_compare["pct_perdida"] = plan_compare["pct_perdida"] * 100
        plan_compare = plan_compare.sort_values("margen_medio")

        chart_plan = plan_compare[[plan_type_col, "prima_media", "coste_medio"]].set_index(plan_type_col)
        st.bar_chart(chart_plan)
        st.dataframe(
            plan_compare.rename(columns={
                plan_type_col: "tipo_plan",
                "pct_perdida": "%_perdida"
            }).style.format({
                "prima_media": lambda x: fmt_num(x),
                "coste_medio": lambda x: fmt_num(x),
                "margen_medio": lambda x: fmt_num(x),
                "%_perdida": lambda x: fmt_pct(x),
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No existe columna de tipo de plan.")


# =========================================================
# BOLSAS DE PÉRDIDA
# =========================================================
st.markdown("## Bolsas de pérdida más relevantes")

group_candidates = [
    ("Tipo de plan", plan_type_col),
    ("Tier del plan", plan_tier_col),
    ("Cobertura", coverage_scope_col),
    ("Red de proveedores", network_col),
]

selected_group_label = st.selectbox(
    "Agrupar tabla por",
    options=[label for label, col in group_candidates if col is not None],
    index=0
)

selected_group_col = dict(group_candidates)[selected_group_label]

agg_dict = {
    "member_id": "count",
    "premium_annual": "mean",
    "approved_cost_sum": "mean",
    "margen_tecnico": "mean",
    "perdida_tecnica_flag": "mean",
    "severidad_perdida": "sum",
}

if risk_prob_col:
    agg_dict[risk_prob_col] = "mean"
if claims_col:
    agg_dict[claims_col] = "mean"
if abuse_score_col:
    agg_dict[abuse_score_col] = "mean"
if chronic_count_col:
    agg_dict[chronic_count_col] = "mean"

loss_pockets = (
    df_filtered
    .groupby(selected_group_col, dropna=False)
    .agg(agg_dict)
    .reset_index()
    .rename(columns={
        selected_group_col: "grupo",
        "member_id": "registros",
        "premium_annual": "prima_media",
        "approved_cost_sum": "coste_medio",
        "margen_tecnico": "margen_medio",
        "perdida_tecnica_flag": "pct_perdida",
        "severidad_perdida": "perdida_total_abs",
        risk_prob_col: "riesgo_medio" if risk_prob_col else "riesgo_medio",
        claims_col: "claims_medios" if claims_col else "claims_medios",
        abuse_score_col: "abuse_score_medio" if abuse_score_col else "abuse_score_medio",
        chronic_count_col: "cronicos_medios" if chronic_count_col else "cronicos_medios",
    })
)

loss_pockets["pct_perdida"] = loss_pockets["pct_perdida"] * 100
loss_pockets["ratio_coste_prima_medio"] = np.where(
    loss_pockets["prima_media"] > 0,
    loss_pockets["coste_medio"] / loss_pockets["prima_media"],
    np.nan
)
loss_pockets = loss_pockets.sort_values(
    ["pct_perdida", "margen_medio"],
    ascending=[False, True]
)

style_map = {
    "prima_media": lambda x: fmt_num(x),
    "coste_medio": lambda x: fmt_num(x),
    "margen_medio": lambda x: fmt_num(x),
    "pct_perdida": lambda x: fmt_pct(x),
    "perdida_total_abs": lambda x: fmt_num(x),
    "ratio_coste_prima_medio": lambda x: fmt_num(x),
}
for optional_col in ["riesgo_medio", "claims_medios", "abuse_score_medio", "cronicos_medios"]:
    if optional_col in loss_pockets.columns:
        style_map[optional_col] = lambda x: fmt_num(x)

st.dataframe(
    loss_pockets.style.format(style_map),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# CONCENTRACIÓN DEL COSTE
# =========================================================
st.markdown("## Concentración del coste")

conc1, conc2 = st.columns(2)

with conc1:
    st.markdown("### Coste total por segmento de riesgo")
    if risk_segment_col:
        cost_by_segment = (
            df_filtered
            .groupby(risk_segment_col, dropna=False)
            .agg(
                coste_total=("approved_cost_sum", "sum"),
                prima_total=("premium_annual", "sum"),
                perdida_total=("severidad_perdida", "sum")
            )
            .reset_index()
        )
        cost_by_segment["orden"] = cost_by_segment[risk_segment_col].map(segment_order_key)
        cost_by_segment = cost_by_segment.sort_values("orden").drop(columns="orden")
        st.bar_chart(cost_by_segment.set_index(risk_segment_col)[["coste_total", "prima_total"]])
        st.dataframe(
            cost_by_segment.rename(columns={risk_segment_col: "segmento_riesgo"}).style.format({
                "coste_total": lambda x: fmt_num(x),
                "prima_total": lambda x: fmt_num(x),
                "perdida_total": lambda x: fmt_num(x),
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No existe columna de segmento de riesgo.")

with conc2:
    st.markdown("### Concentración en el 10% de miembros más costosos")
    if n_registros > 0:
        top_n = max(1, int(len(df_filtered) * 0.10))
        top_cost = (
            df_filtered
            .sort_values("approved_cost_sum", ascending=False)
            .head(top_n)
        )

        pct_cost_top10 = (
            top_cost["approved_cost_sum"].sum() / coste_total * 100
            if coste_total > 0 else np.nan
        )
        pct_loss_top10 = (
            top_cost["severidad_perdida"].sum() / df_filtered["severidad_perdida"].sum() * 100
            if df_filtered["severidad_perdida"].sum() > 0 else np.nan
        )

        st.metric("Peso del top 10% en coste total", fmt_pct(pct_cost_top10))
        st.metric("Peso del top 10% en pérdida total", fmt_pct(pct_loss_top10))

        top10_resume = pd.DataFrame({
            "indicador": ["Miembros en top 10%", "% del coste total", "% de la pérdida total"],
            "valor": [fmt_int(top_n), fmt_pct(pct_cost_top10), fmt_pct(pct_loss_top10)]
        })
        st.dataframe(top10_resume, use_container_width=True, hide_index=True)
    else:
        st.info("No hay datos suficientes.")


# =========================================================
# CASOS PRIORITARIOS
# =========================================================
st.markdown("## Casos prioritarios para revisión")

priority_sort_cols = ["margen_tecnico", "approved_cost_sum"]
ascending_flags = [True, False]

priority_df = df_filtered.copy().sort_values(priority_sort_cols, ascending=ascending_flags)

priority_cols = [
    "member_id",
    "policy_id",
    risk_segment_col,
    risk_prob_col,
    plan_type_col,
    coverage_scope_col,
    "premium_annual",
    "approved_cost_sum",
    "margen_tecnico",
    "ratio_coste_prima",
    claims_col,
    chronic_flag_col,
    chronic_count_col,
    abuse_score_col,
    flagged_cost_col,
]
priority_cols = [c for c in priority_cols if c and c in priority_df.columns]

st.caption(
    "Se priorizan los registros con peor margen técnico y alto coste aprobado para facilitar revisión de pricing, utilización o abuso."
)
st.dataframe(
    priority_df[priority_cols].head(25),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# VISTA DETALLADA
# =========================================================
st.markdown("## Vista detallada póliza-miembro")

detail_cols = [
    "member_id",
    "policy_id",
    risk_segment_col,
    risk_prob_col,
    plan_type_col,
    plan_tier_col,
    coverage_scope_col,
    network_col,
    "premium_annual",
    "approved_cost_sum",
    "margen_tecnico",
    "ratio_coste_prima",
    claims_col,
    chronic_flag_col,
    chronic_count_col,
    abuse_score_col,
    flagged_cost_col,
    pricing_adequacy_col,
]
detail_cols = [c for c in detail_cols if c and c in df_filtered.columns]

st.dataframe(
    df_filtered[detail_cols],
    use_container_width=True,
    hide_index=True
)
