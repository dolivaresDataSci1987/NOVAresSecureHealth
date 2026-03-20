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
    page_title="Pricing y adecuación de póliza",
    page_icon="💰",
    layout="wide"
)


# =========================================================
# DATA
# =========================================================
@st.cache_data
def get_data():
    return load_policy_member_master()


df = get_data().copy()

st.title("💰 Pricing y adecuación de póliza")
st.caption(
    "Evaluación de si la prima actual es adecuada para el nivel de riesgo, coste esperado y uso observado del asegurado."
)


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


def normalize_severity(x):
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


def severity_order_key(x):
    order = {
        "Muy bajo": 1,
        "Bajo": 2,
        "Medio": 3,
        "Alto": 4,
        "Muy alto": 5,
        "Sin dato": 99,
    }
    return order.get(x, 50)


def first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def render_info_box(title, text):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(text)


def build_dictionary_df(dictionary_map):
    return pd.DataFrame(
        [{"término": k, "explicación": v} for k, v in dictionary_map.items()]
    )


def recommend_action(row):
    margen = row.get("margen_estimado", np.nan)
    ratio = row.get("ratio_prima_coste", np.nan)
    riesgo = row.get("_risk_prob", np.nan)
    abuso_score = row.get("_abuse_score", np.nan)
    fraude_max = row.get("_provider_fraud_max", np.nan)

    if pd.notna(abuso_score) and abuso_score >= 0.75:
        return "Revisión manual por posible abuso"

    if pd.notna(fraude_max) and fraude_max >= 0.75:
        return "Revisión manual por exposición a proveedor riesgoso"

    if pd.notna(margen):
        if margen >= 0 and pd.notna(ratio) and ratio >= 1.20:
            return "Mantener plan actual"
        if margen >= 0 and pd.notna(ratio) and ratio < 1.20:
            return "Mantener y monitorizar"
        if margen < 0 and margen >= -50:
            return "Revisar prima"
        if margen < -50 and pd.notna(riesgo) and riesgo >= 0.70:
            return "Migrar a plan más adecuado"
        if margen < -50:
            return "Revisar prima o cobertura"

    return "Revisión comercial"


def explain_recommendation(row):
    margen = row.get("margen_estimado", np.nan)
    ratio = row.get("ratio_prima_coste", np.nan)
    riesgo = row.get("_risk_prob", np.nan)
    abuso_score = row.get("_abuse_score", np.nan)
    fraude_max = row.get("_provider_fraud_max", np.nan)

    parts = []

    if pd.notna(margen):
        if margen >= 0:
            parts.append("la póliza genera margen positivo")
        else:
            parts.append("la póliza presenta margen negativo")

    if pd.notna(ratio):
        if ratio < 1:
            parts.append("la prima actual no cubre el coste estimado")
        elif ratio < 1.2:
            parts.append("la cobertura económica es ajustada")
        else:
            parts.append("la prima cubre holgadamente el coste estimado")

    if pd.notna(riesgo) and riesgo >= 0.70:
        parts.append("el riesgo estimado del cliente es alto")

    if pd.notna(abuso_score) and abuso_score >= 0.75:
        parts.append("hay una señal elevada de posible abuso")

    if pd.notna(fraude_max) and fraude_max >= 0.75:
        parts.append("existe exposición a proveedores con alto riesgo")

    if not parts:
        return "Recomendación basada en la combinación general de prima, coste y riesgo."

    text = ", ".join(parts)
    return text[:1].upper() + text[1:] + "."


def classify_pricing_status(row):
    margen = row.get("margen_estimado", np.nan)
    ratio = row.get("ratio_prima_coste", np.nan)

    if pd.isna(margen) and pd.isna(ratio):
        return "Sin dato"

    if pd.notna(ratio):
        if ratio >= 1.20:
            return "Adecuado"
        if 1.00 <= ratio < 1.20:
            return "Margen bajo"
        if 0.85 <= ratio < 1.00:
            return "Infraprecificado"
        if ratio < 0.85:
            return "Pérdida clara"

    if pd.notna(margen):
        if margen >= 50:
            return "Adecuado"
        if 0 <= margen < 50:
            return "Margen bajo"
        if -50 <= margen < 0:
            return "Infraprecificado"
        if margen < -50:
            return "Pérdida clara"

    return "Sin dato"


def status_order_key(x):
    order = {
        "Adecuado": 1,
        "Margen bajo": 2,
        "Infraprecificado": 3,
        "Pérdida clara": 4,
        "Sin dato": 99,
    }
    return order.get(x, 50)


# =========================================================
# DETECCIÓN DE COLUMNAS
# =========================================================
columns = df.columns.tolist()

member_id_col = first_existing(columns, ["member_id", "customer_id", "insured_id"])
policy_id_col = first_existing(columns, ["policy_id", "policy_number", "policy_code"])
plan_type_col = first_existing(columns, ["plan_type", "plan_name", "product_name"])
coverage_scope_col = first_existing(columns, ["coverage_scope", "coverage_type", "coverage_level"])

premium_col = first_existing(columns, [
    "premium_monthly",
    "monthly_premium",
    "current_premium",
    "premium_amount",
    "premium",
    "policy_premium"
])

expected_cost_col = first_existing(columns, [
    "expected_cost",
    "expected_cost_annual",
    "expected_cost_monthly",
    "predicted_cost",
    "projected_cost",
    "expected_claim_cost"
])

approved_cost_col = first_existing(columns, [
    "approved_cost_sum",
    "observed_cost",
    "total_cost",
    "claims_cost_sum"
])

pricing_score_col = first_existing(columns, [
    "pricing_score",
    "price_adequacy_score",
    "premium_adequacy_score"
])

pricing_status_col = first_existing(columns, [
    "pricing_status",
    "pricing_segment",
    "premium_status"
])

suggested_plan_col = first_existing(columns, [
    "suggested_plan_type",
    "recommended_plan",
    "proposed_plan",
    "next_best_plan"
])

risk_prob_col = first_existing(columns, [
    "predicted_risk_probability",
    "risk_probability",
    "risk_prob"
])

risk_segment_col = first_existing(columns, [
    "predicted_risk_segment",
    "risk_segment"
])

abuse_score_col = first_existing(columns, [
    "member_abuse_score",
    "abuse_score"
])

provider_fraud_max_col = first_existing(columns, [
    "provider_fraud_score_max",
    "max_provider_fraud_score"
])

claims_col = first_existing(columns, [
    "claims_count",
    "claims_n"
])


# =========================================================
# NORMALIZACIÓN
# =========================================================
for col in [
    premium_col,
    expected_cost_col,
    approved_cost_col,
    pricing_score_col,
    risk_prob_col,
    abuse_score_col,
    provider_fraud_max_col,
    claims_col,
]:
    if col:
        df[col] = to_numeric(df[col])

if risk_segment_col:
    df[risk_segment_col] = df[risk_segment_col].apply(normalize_severity)

if pricing_status_col:
    df[pricing_status_col] = df[pricing_status_col].astype(str).fillna("Sin dato")


# =========================================================
# VARIABLES DERIVADAS
# =========================================================
if premium_col:
    df["prima_actual"] = df[premium_col]
else:
    df["prima_actual"] = np.nan

# prioridad de coste: esperado > observado
if expected_cost_col:
    df["coste_referencia"] = df[expected_cost_col]
elif approved_cost_col:
    df["coste_referencia"] = df[approved_cost_col]
else:
    df["coste_referencia"] = np.nan

df["margen_estimado"] = df["prima_actual"] - df["coste_referencia"]

df["ratio_prima_coste"] = np.where(
    df["coste_referencia"] > 0,
    df["prima_actual"] / df["coste_referencia"],
    np.nan
)

df["_risk_prob"] = df[risk_prob_col] if risk_prob_col else np.nan
df["_abuse_score"] = df[abuse_score_col] if abuse_score_col else np.nan
df["_provider_fraud_max"] = df[provider_fraud_max_col] if provider_fraud_max_col else np.nan

if pricing_status_col:
    df["estado_pricing"] = df[pricing_status_col].replace(
        {
            "adequate": "Adecuado",
            "underpriced": "Infraprecificado",
            "overpriced": "Sobreprecio",
            "low_margin": "Margen bajo",
            "loss": "Pérdida clara",
        }
    )
else:
    df["estado_pricing"] = df.apply(classify_pricing_status, axis=1)

df["estado_pricing"] = df["estado_pricing"].fillna("Sin dato").astype(str)

df["rentable_flag"] = df["margen_estimado"] >= 0
df["perdida_flag"] = df["margen_estimado"] < 0

df["recomendacion_pricing"] = df.apply(recommend_action, axis=1)
df["justificacion_recomendacion"] = df.apply(explain_recommendation, axis=1)

if suggested_plan_col:
    df["plan_sugerido_final"] = np.where(
        df[suggested_plan_col].notna() & (df[suggested_plan_col].astype(str).str.strip() != ""),
        df[suggested_plan_col].astype(str),
        "Mantener plan actual"
    )
else:
    df["plan_sugerido_final"] = np.where(
        df["recomendacion_pricing"].isin(["Migrar a plan más adecuado"]),
        "Revisar alternativa de plan",
        "Mantener plan actual"
    )


# =========================================================
# DICCIONARIO
# =========================================================
dictionary_map = {
    "Prima actual": "Importe que actualmente paga el cliente por su póliza o plan.",
    "Coste de referencia": "Coste esperado o, si no existe, coste observado/aprobado usado como referencia económica.",
    "Margen estimado": "Diferencia entre prima actual y coste de referencia. Si es negativo, la póliza destruye margen.",
    "Ratio prima / coste": "Relación entre lo que se cobra y lo que cuesta el cliente. Cuanto más alto, mayor holgura económica.",
    "Estado de pricing": "Clasificación operativa de la adecuación del precio actual.",
    "Adecuado": "La prima parece suficiente para cubrir el coste y dejar margen razonable.",
    "Margen bajo": "La póliza aún no está en pérdida, pero el colchón económico es reducido.",
    "Infraprecificado": "La prima está por debajo de lo deseable para el coste estimado del cliente.",
    "Pérdida clara": "El cliente está generando margen claramente negativo.",
    "Recomendación de pricing": "Sugerencia comercial o técnica sobre mantener, revisar o cambiar la póliza.",
    "Plan sugerido": "Tipo de plan alternativo recomendado, cuando existe esa columna en el dataset.",
}


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Filtros · Pricing")

df_filtered = df.copy()

status_options = sorted(
    df_filtered["estado_pricing"].dropna().astype(str).unique().tolist(),
    key=status_order_key
)
selected_status = st.sidebar.multiselect(
    "Estado de pricing",
    options=status_options,
    default=status_options
)
if selected_status:
    df_filtered = df_filtered[df_filtered["estado_pricing"].astype(str).isin(selected_status)]

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

only_loss_cases = st.sidebar.checkbox("Mostrar solo pólizas en pérdida", value=False)
if only_loss_cases:
    df_filtered = df_filtered[df_filtered["perdida_flag"]]

search_member = st.sidebar.text_input("Buscar member_id")
if search_member and member_id_col:
    df_filtered = df_filtered[
        df_filtered[member_id_col].astype(str).str.contains(search_member, case=False, na=False)
    ]

search_policy = st.sidebar.text_input("Buscar policy_id")
if search_policy and policy_id_col:
    df_filtered = df_filtered[
        df_filtered[policy_id_col].astype(str).str.contains(search_policy, case=False, na=False)
    ]

if df_filtered["ratio_prima_coste"].notna().any():
    ratio_min = float(np.nanmin(df_filtered["ratio_prima_coste"]))
    ratio_max = float(np.nanmax(df_filtered["ratio_prima_coste"]))
    if ratio_min < ratio_max:
        ratio_range = st.sidebar.slider(
            "Rango ratio prima / coste",
            min_value=float(ratio_min),
            max_value=float(ratio_max),
            value=(float(ratio_min), float(ratio_max))
        )
        df_filtered = df_filtered[
            df_filtered["ratio_prima_coste"].between(ratio_range[0], ratio_range[1], inclusive="both")
        ]


# =========================================================
# KPIS
# =========================================================
n_rows = len(df_filtered)

rentables_n = int(df_filtered["rentable_flag"].sum()) if n_rows > 0 else 0
rentables_pct = (rentables_n / n_rows * 100) if n_rows > 0 else np.nan

perdida_n = int(df_filtered["perdida_flag"].sum()) if n_rows > 0 else 0
perdida_pct = (perdida_n / n_rows * 100) if n_rows > 0 else np.nan

prima_media = df_filtered["prima_actual"].mean()
coste_medio = df_filtered["coste_referencia"].mean()
margen_medio = df_filtered["margen_estimado"].mean()

prima_total = df_filtered["prima_actual"].sum()
coste_total = df_filtered["coste_referencia"].sum()
margen_total = df_filtered["margen_estimado"].sum()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Clientes analizados", fmt_int(n_rows))
k2.metric("Pólizas rentables", fmt_int(rentables_n), delta=fmt_pct(rentables_pct) if pd.notna(rentables_pct) else None)
k3.metric("Pólizas en pérdida", fmt_int(perdida_n), delta=fmt_pct(perdida_pct) if pd.notna(perdida_pct) else None)
k4.metric("Prima media actual", fmt_num(prima_media))
k5.metric("Coste medio de referencia", fmt_num(coste_medio))
k6.metric("Margen medio estimado", fmt_num(margen_medio))


# =========================================================
# LECTURA DE NEGOCIO
# =========================================================
insights = []

if n_rows > 0:
    insights.append(f"La vista actual incluye **{fmt_int(n_rows)} pólizas o clientes** analizados.")

if pd.notna(rentables_pct):
    insights.append(
        f"El **{fmt_pct(rentables_pct)}** parece estar en zona rentable, mientras que el **{fmt_pct(perdida_pct)}** presenta margen negativo."
    )

if pd.notna(margen_total):
    if margen_total >= 0:
        insights.append(
            f"El margen agregado estimado de la vista filtrada es **positivo** y asciende a **{fmt_num(margen_total)}**."
        )
    else:
        insights.append(
            f"El margen agregado estimado de la vista filtrada es **negativo** y asciende a **{fmt_num(margen_total)}**, lo que sugiere un problema de pricing o segmentación de producto."
        )

if pd.notna(prima_media) and pd.notna(coste_medio):
    insights.append(
        f"La comparación entre **prima media actual ({fmt_num(prima_media)})** y **coste medio de referencia ({fmt_num(coste_medio)})** ayuda a valorar si el problema es estructural o se concentra en ciertos segmentos."
    )

with st.container(border=True):
    st.markdown("### Lectura de negocio")
    for item in insights:
        st.markdown(f"- {item}")


# =========================================================
# DICCIONARIO
# =========================================================
with st.expander("Diccionario de métricas de pricing", expanded=False):
    st.dataframe(
        build_dictionary_df(dictionary_map),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# BLOQUES GENERALES
# =========================================================
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Distribución por estado de pricing")
    status_dist = (
        df_filtered["estado_pricing"]
        .fillna("Sin dato")
        .value_counts()
        .rename_axis("estado_pricing")
        .reset_index(name="casos")
    )
    status_dist["orden"] = status_dist["estado_pricing"].map(status_order_key)
    status_dist = status_dist.sort_values("orden").drop(columns="orden")
    st.bar_chart(status_dist.set_index("estado_pricing"))
    render_info_box(
        "Cómo interpretar este gráfico",
        "Permite ver qué parte de la cartera está correctamente tarificada y qué parte requiere intervención. "
        "Una concentración en **Infraprecificado** o **Pérdida clara** indica necesidad de ajuste de precio, rediseño de cobertura o migración a otro plan."
    )
    st.dataframe(status_dist, use_container_width=True, hide_index=True)

with c2:
    st.markdown("### Margen estimado por estado de pricing")
    margin_by_status = (
        df_filtered.groupby("estado_pricing", dropna=False)
        .agg(
            clientes=(member_id_col, "count") if member_id_col else ("estado_pricing", "count"),
            prima_media=("prima_actual", "mean"),
            coste_medio=("coste_referencia", "mean"),
            margen_medio=("margen_estimado", "mean")
        )
        .reset_index()
    )
    margin_by_status["orden"] = margin_by_status["estado_pricing"].map(status_order_key)
    margin_by_status = margin_by_status.sort_values("orden").drop(columns="orden")
    st.bar_chart(margin_by_status.set_index("estado_pricing")[["margen_medio"]])
    render_info_box(
        "Cómo interpretar este gráfico",
        "Muestra si los distintos estados de pricing tienen realmente traducción económica. "
        "Sirve para validar que los grupos clasificados como problemáticos son también los que concentran peor margen."
    )
    st.dataframe(
        margin_by_status.style.format({
            "prima_media": lambda x: fmt_num(x),
            "coste_medio": lambda x: fmt_num(x),
            "margen_medio": lambda x: fmt_num(x),
        }),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# BOLSAS DE INFRAPRICE
# =========================================================
st.markdown("### Bolsas de infrapricing por segmento")

group_candidates = []
if plan_type_col:
    group_candidates.append(("Tipo de plan", plan_type_col))
if coverage_scope_col:
    group_candidates.append(("Cobertura", coverage_scope_col))
if risk_segment_col:
    group_candidates.append(("Segmento de riesgo", risk_segment_col))
if pricing_status_col:
    group_candidates.append(("Estado de pricing original", pricing_status_col))

if group_candidates:
    selected_group_label = st.selectbox(
        "Agrupar análisis por",
        options=[label for label, _ in group_candidates],
        index=0
    )
    selected_group_col = dict(group_candidates)[selected_group_label]

    group_summary = (
        df_filtered.groupby(selected_group_col, dropna=False)
        .agg(
            clientes=(member_id_col, "count") if member_id_col else (selected_group_col, "count"),
            prima_media=("prima_actual", "mean"),
            coste_medio=("coste_referencia", "mean"),
            margen_medio=("margen_estimado", "mean"),
            pct_perdida=("perdida_flag", "mean"),
            ratio_medio=("ratio_prima_coste", "mean"),
        )
        .reset_index()
        .rename(columns={selected_group_col: "grupo"})
    )

    group_summary["pct_perdida"] = group_summary["pct_perdida"] * 100
    group_summary = group_summary.sort_values(
        ["pct_perdida", "margen_medio"],
        ascending=[False, True]
    )

    render_info_box(
        "Cómo interpretar esta tabla",
        "Esta tabla ayuda a detectar **dónde** se concentra el problema de pricing. "
        "Los grupos con mayor porcentaje de pólizas en pérdida, menor ratio prima/coste y peor margen medio son candidatos prioritarios para rediseño comercial o técnico."
    )

    st.dataframe(
        group_summary.style.format({
            "prima_media": lambda x: fmt_num(x),
            "coste_medio": lambda x: fmt_num(x),
            "margen_medio": lambda x: fmt_num(x),
            "pct_perdida": lambda x: fmt_pct(x),
            "ratio_medio": lambda x: fmt_num(x),
        }),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No hay columnas suficientes para agrupar el análisis por segmentos.")


# =========================================================
# RENTABLES VS NO RENTABLES
# =========================================================
st.markdown("### Cartera rentable vs cartera no rentable")

rv1, rv2 = st.columns(2)

with rv1:
    rentable_summary = pd.DataFrame({
        "grupo": ["Rentables", "En pérdida"],
        "casos": [rentables_n, perdida_n]
    })
    st.bar_chart(rentable_summary.set_index("grupo"))
    render_info_box(
        "Cómo interpretar este bloque",
        "Resume el equilibrio entre pólizas que sostienen margen y pólizas que lo erosionan. "
        "Es una lectura rápida para saber si el problema es puntual o relevante dentro de la cartera filtrada."
    )
    st.dataframe(rentable_summary, use_container_width=True, hide_index=True)

with rv2:
    portfolio_summary = pd.DataFrame({
        "métrica": ["Prima total", "Coste total de referencia", "Margen total estimado"],
        "valor": [prima_total, coste_total, margen_total]
    })
    st.dataframe(
        portfolio_summary.style.format({"valor": lambda x: fmt_num(x)}),
        use_container_width=True,
        hide_index=True
    )
    render_info_box(
        "Lectura económica agregada",
        "La comparación entre ingreso estimado por primas y coste de referencia permite valorar si la vista seleccionada destruye o crea valor económico."
    )


# =========================================================
# PANEL DE RECOMENDACIÓN CLIENTE A CLIENTE
# =========================================================
st.markdown("### Panel de análisis cliente a cliente")

st.caption(
    "Este bloque permite priorizar clientes para decisión comercial, revisión de prima o posible cambio de plan."
)

sort_candidates = []
ascending = []

if "margen_estimado" in df_filtered.columns:
    sort_candidates.append("margen_estimado")
    ascending.append(True)

if "ratio_prima_coste" in df_filtered.columns:
    sort_candidates.append("ratio_prima_coste")
    ascending.append(True)

if risk_prob_col:
    sort_candidates.append(risk_prob_col)
    ascending.append(False)

review_df = df_filtered.sort_values(sort_candidates, ascending=ascending).copy() if sort_candidates else df_filtered.copy()

rename_map = {
    member_id_col: "member_id" if member_id_col else None,
    policy_id_col: "policy_id" if policy_id_col else None,
    plan_type_col: "tipo_plan" if plan_type_col else None,
    coverage_scope_col: "cobertura" if coverage_scope_col else None,
    risk_segment_col: "segmento_riesgo" if risk_segment_col else None,
    risk_prob_col: "prob_riesgo" if risk_prob_col else None,
    claims_col: "claims" if claims_col else None,
}
rename_map = {k: v for k, v in rename_map.items() if k}

panel_cols = [
    member_id_col,
    policy_id_col,
    plan_type_col,
    coverage_scope_col,
    risk_segment_col,
    risk_prob_col,
    claims_col,
    "prima_actual",
    "coste_referencia",
    "margen_estimado",
    "ratio_prima_coste",
    "estado_pricing",
    "recomendacion_pricing",
    "plan_sugerido_final",
    "justificacion_recomendacion",
]
panel_cols = [c for c in panel_cols if c in review_df.columns]

panel_view = review_df[panel_cols].rename(columns=rename_map)

render_info_box(
    "Cómo interpretar esta tabla",
    "Los primeros casos son los más prioritarios porque combinan peor margen, menor cobertura económica de la prima y, cuando existe, mayor riesgo. "
    "La recomendación no sustituye la decisión final de underwriting o negocio, pero ayuda a ordenar la revisión."
)

st.dataframe(
    panel_view.head(50),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# FICHA INDIVIDUAL
# =========================================================
st.markdown("### Ficha individual de cliente / póliza")

selector_options = []

if member_id_col and policy_id_col:
    selector_options = (
        review_df[[member_id_col, policy_id_col]]
        .fillna("Sin dato")
        .astype(str)
        .apply(lambda x: f"{x[member_id_col]} | {x[policy_id_col]}", axis=1)
        .tolist()
    )
elif member_id_col:
    selector_options = review_df[member_id_col].fillna("Sin dato").astype(str).tolist()
elif policy_id_col:
    selector_options = review_df[policy_id_col].fillna("Sin dato").astype(str).tolist()

if selector_options:
    selected_case = st.selectbox(
        "Seleccionar caso",
        options=selector_options,
        index=0
    )

    if member_id_col and policy_id_col:
        member_val, policy_val = [x.strip() for x in selected_case.split("|", 1)]
        case_df = review_df[
            (review_df[member_id_col].astype(str) == member_val) &
            (review_df[policy_id_col].astype(str) == policy_val)
        ]
    elif member_id_col:
        case_df = review_df[review_df[member_id_col].astype(str) == selected_case]
    else:
        case_df = review_df[review_df[policy_id_col].astype(str) == selected_case]

    if not case_df.empty:
        row = case_df.iloc[0]

        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Prima actual", fmt_num(row.get("prima_actual", np.nan)))
        f2.metric("Coste de referencia", fmt_num(row.get("coste_referencia", np.nan)))
        f3.metric("Margen estimado", fmt_num(row.get("margen_estimado", np.nan)))
        f4.metric("Ratio prima / coste", fmt_num(row.get("ratio_prima_coste", np.nan)))

        with st.container(border=True):
            st.markdown("### Lectura individual")
            if plan_type_col:
                st.markdown(f"- **Plan actual:** {row.get(plan_type_col, 'Sin dato')}")
            if coverage_scope_col:
                st.markdown(f"- **Cobertura:** {row.get(coverage_scope_col, 'Sin dato')}")
            st.markdown(f"- **Estado de pricing:** {row.get('estado_pricing', 'Sin dato')}")
            st.markdown(f"- **Recomendación:** {row.get('recomendacion_pricing', 'Sin dato')}")
            st.markdown(f"- **Plan sugerido:** {row.get('plan_sugerido_final', 'Sin dato')}")
            st.markdown(f"- **Justificación:** {row.get('justificacion_recomendacion', 'Sin dato')}")

            if risk_prob_col and pd.notna(row.get(risk_prob_col, np.nan)):
                st.markdown(f"- **Probabilidad de riesgo:** {fmt_num(row.get(risk_prob_col, np.nan))}")

            if claims_col and pd.notna(row.get(claims_col, np.nan)):
                st.markdown(f"- **Claims:** {fmt_num(row.get(claims_col, np.nan))}")

            if abuse_score_col and pd.notna(row.get(abuse_score_col, np.nan)):
                st.markdown(f"- **Abuse score:** {fmt_num(row.get(abuse_score_col, np.nan))}")

            if provider_fraud_max_col and pd.notna(row.get(provider_fraud_max_col, np.nan)):
                st.markdown(f"- **Máx. fraude proveedor vinculado:** {fmt_num(row.get(provider_fraud_max_col, np.nan))}")
else:
    st.info("No hay identificadores suficientes para construir una ficha individual.")


# =========================================================
# VISTA DETALLADA COMPLETA
# =========================================================
st.markdown("### Vista detallada completa")

full_cols = [
    member_id_col,
    policy_id_col,
    plan_type_col,
    coverage_scope_col,
    risk_segment_col,
    risk_prob_col,
    claims_col,
    premium_col,
    expected_cost_col,
    approved_cost_col,
    pricing_score_col,
    pricing_status_col,
    suggested_plan_col,
    abuse_score_col,
    provider_fraud_max_col,
    "prima_actual",
    "coste_referencia",
    "margen_estimado",
    "ratio_prima_coste",
    "estado_pricing",
    "recomendacion_pricing",
    "plan_sugerido_final",
    "justificacion_recomendacion",
]
full_cols = [c for c in full_cols if c in review_df.columns]

st.dataframe(
    review_df[full_cols],
    use_container_width=True,
    hide_index=True
)
