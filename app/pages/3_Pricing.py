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
    "Evaluación de si la prima actual es adecuada para el coste esperado del cliente, "
    "identificando pólizas rentables, pólizas en pérdida y posibles acciones de ajuste comercial."
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


def normalize_text(x):
    if pd.isna(x):
        return "Sin dato"
    txt = str(x).strip()
    return txt if txt else "Sin dato"


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


def first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def build_dictionary_df(dictionary_map):
    return pd.DataFrame(
        [{"término": k, "explicación": v} for k, v in dictionary_map.items()]
    )


def render_info_box(title, text):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(text)


def status_order_key(x):
    order = {
        "Rentable": 1,
        "Margen bajo": 2,
        "Infraprecificado": 3,
        "Pérdida clara": 4,
        "Sin dato": 99,
    }
    return order.get(x, 50)


def get_margin_tolerance(cost_value):
    if pd.isna(cost_value):
        return 15.0
    return max(15.0, 0.05 * float(cost_value))


def classify_pricing_status(row):
    prima = row.get("prima_actual", np.nan)
    coste = row.get("coste_referencia", np.nan)
    margen = row.get("margen_estimado", np.nan)
    ratio = row.get("ratio_prima_coste", np.nan)

    if pd.isna(prima) or pd.isna(coste):
        return "Sin dato"

    tol = get_margin_tolerance(coste)

    if pd.notna(margen):
        if margen >= tol and pd.notna(ratio) and ratio >= 1.08:
            return "Rentable"
        if -tol < margen < tol:
            return "Margen bajo"
        if margen <= -tol and margen > -(2 * tol):
            return "Infraprecificado"
        if margen <= -(2 * tol):
            return "Pérdida clara"

    if pd.notna(ratio):
        if ratio >= 1.08:
            return "Rentable"
        if 0.95 <= ratio < 1.08:
            return "Margen bajo"
        if 0.85 <= ratio < 0.95:
            return "Infraprecificado"
        if ratio < 0.85:
            return "Pérdida clara"

    return "Sin dato"


def recommend_action(row):
    estado = row.get("estado_pricing", "Sin dato")
    riesgo = row.get("_risk_prob", np.nan)
    abuso = row.get("_abuse_score", np.nan)
    fraude = row.get("_provider_fraud_max", np.nan)

    if pd.notna(abuso) and abuso >= 0.75:
        return "Revisión manual por abuso"
    if pd.notna(fraude) and fraude >= 0.75:
        return "Revisión manual por proveedor riesgoso"

    if estado == "Rentable":
        return "Mantener plan actual"
    if estado == "Margen bajo":
        return "Mantener y monitorizar"
    if estado == "Infraprecificado":
        return "Revisar prima"
    if estado == "Pérdida clara" and pd.notna(riesgo) and riesgo >= 0.70:
        return "Migrar a plan más adecuado"
    if estado == "Pérdida clara":
        return "Revisar prima o cobertura"

    return "Revisión comercial"


def explain_recommendation(row):
    estado = row.get("estado_pricing", "Sin dato")
    margen = row.get("margen_estimado", np.nan)
    ratio = row.get("ratio_prima_coste", np.nan)
    riesgo = row.get("_risk_prob", np.nan)

    parts = [f"Estado de pricing: {estado.lower()}"]

    if pd.notna(margen):
        if margen >= 0:
            parts.append(f"margen estimado positivo ({fmt_num(margen)})")
        else:
            parts.append(f"margen estimado negativo ({fmt_num(margen)})")

    if pd.notna(ratio):
        parts.append(f"ratio prima/coste de {fmt_num(ratio)}")

    if pd.notna(riesgo) and riesgo >= 0.70:
        parts.append("riesgo elevado")

    text = ", ".join(parts)
    return text[:1].upper() + text[1:] + "."


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
    df[pricing_status_col] = df[pricing_status_col].apply(normalize_text)


# =========================================================
# VARIABLES DERIVADAS
# =========================================================
df["prima_actual"] = df[premium_col] if premium_col else np.nan

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

# Estado operativo recalculado, sin depender de que la columna original venga rara
df["estado_pricing"] = df.apply(classify_pricing_status, axis=1)

# Conservamos la original solo como referencia, no como eje principal
if pricing_status_col:
    df["estado_pricing_original"] = df[pricing_status_col]
else:
    df["estado_pricing_original"] = "Sin dato"

df["rentable_flag"] = df["estado_pricing"] == "Rentable"
df["margen_bajo_flag"] = df["estado_pricing"] == "Margen bajo"
df["infrapricing_flag"] = df["estado_pricing"].isin(["Infraprecificado", "Pérdida clara"])
df["perdida_clara_flag"] = df["estado_pricing"] == "Pérdida clara"

df["ganancia_bruta"] = np.where(df["margen_estimado"] > 0, df["margen_estimado"], 0.0)
df["perdida_bruta"] = np.where(df["margen_estimado"] < 0, -df["margen_estimado"], 0.0)

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
    "Prima actual": "Importe que actualmente paga el cliente por su póliza.",
    "Coste de referencia": "Coste esperado del cliente. Si no existe, se usa el coste observado/aprobado.",
    "Margen estimado": "Prima actual menos coste de referencia.",
    "Ratio prima / coste": "Relación entre lo que se cobra y lo que cuesta el cliente.",
    "Rentable": "La prima cubre el coste con margen razonable.",
    "Margen bajo": "La póliza no está claramente en pérdida, pero el colchón económico es pequeño.",
    "Infraprecificado": "La prima se queda corta frente al coste esperado.",
    "Pérdida clara": "La póliza presenta un déficit económico más evidente.",
    "Recomendación de pricing": "Acción sugerida para mantener, revisar o cambiar el plan.",
    "Plan sugerido": "Alternativa comercial o técnica propuesta para el cliente.",
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

if risk_segment_col:
    risk_options = sorted(
        df_filtered[risk_segment_col].dropna().astype(str).unique().tolist()
    )
    selected_risk = st.sidebar.multiselect(
        "Segmento de riesgo",
        options=risk_options,
        default=risk_options
    )
    if selected_risk:
        df_filtered = df_filtered[df_filtered[risk_segment_col].astype(str).isin(selected_risk)]

only_problematic = st.sidebar.checkbox("Mostrar solo casos problemáticos", value=False)
if only_problematic:
    df_filtered = df_filtered[df_filtered["estado_pricing"].isin(["Infraprecificado", "Pérdida clara"])]

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


# =========================================================
# KPIS
# =========================================================
n_rows = len(df_filtered)

rentables_n = int(df_filtered["rentable_flag"].sum()) if n_rows > 0 else 0
rentables_pct = (rentables_n / n_rows * 100) if n_rows > 0 else np.nan

margen_bajo_n = int(df_filtered["margen_bajo_flag"].sum()) if n_rows > 0 else 0
margen_bajo_pct = (margen_bajo_n / n_rows * 100) if n_rows > 0 else np.nan

perdida_n = int(df_filtered["perdida_clara_flag"].sum()) if n_rows > 0 else 0
perdida_pct = (perdida_n / n_rows * 100) if n_rows > 0 else np.nan

prima_total = df_filtered["prima_actual"].sum()
coste_total = df_filtered["coste_referencia"].sum()
margen_total = df_filtered["margen_estimado"].sum()
ganancia_total = df_filtered["ganancia_bruta"].sum()
perdida_total = df_filtered["perdida_bruta"].sum()

prima_media = df_filtered["prima_actual"].mean()
coste_medio = df_filtered["coste_referencia"].mean()
margen_medio = df_filtered["margen_estimado"].mean()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Clientes analizados", fmt_int(n_rows))
k2.metric("Rentables", fmt_int(rentables_n), delta=fmt_pct(rentables_pct) if pd.notna(rentables_pct) else None)
k3.metric("Margen bajo", fmt_int(margen_bajo_n), delta=fmt_pct(margen_bajo_pct) if pd.notna(margen_bajo_pct) else None)
k4.metric("Pérdida clara", fmt_int(perdida_n), delta=fmt_pct(perdida_pct) if pd.notna(perdida_pct) else None)
k5.metric("Ganancia total estimada", fmt_num(ganancia_total))
k6.metric("Pérdida total estimada", fmt_num(perdida_total))


# =========================================================
# LECTURA DE NEGOCIO
# =========================================================
insights = []

if n_rows > 0:
    insights.append(f"La vista actual incluye **{fmt_int(n_rows)} clientes o pólizas**.")

if pd.notna(rentables_pct):
    insights.append(
        f"El **{fmt_pct(rentables_pct)}** se encuentra en zona **rentable**, el **{fmt_pct(margen_bajo_pct)}** está en **margen bajo**, "
        f"y el **{fmt_pct(perdida_pct)}** presenta **pérdida clara**."
    )

if pd.notna(margen_total):
    if margen_total >= 0:
        insights.append(
            f"El **margen neto agregado** de la vista filtrada es **positivo ({fmt_num(margen_total)})**."
        )
    else:
        insights.append(
            f"El **margen neto agregado** de la vista filtrada es **negativo ({fmt_num(margen_total)})**, "
            "por lo que la estructura actual de pricing debería revisarse."
        )

if pd.notna(ganancia_total) and pd.notna(perdida_total):
    insights.append(
        f"La cartera visible genera una **ganancia total estimada de {fmt_num(ganancia_total)}** "
        f"y una **pérdida total estimada de {fmt_num(perdida_total)}**."
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
# VISIÓN GENERAL
# =========================================================
g1, g2 = st.columns(2)

with g1:
    st.markdown("### Distribución real por estado de pricing")
    status_dist = (
        df_filtered["estado_pricing"]
        .fillna("Sin dato")
        .value_counts()
        .rename_axis("estado_pricing")
        .reset_index(name="casos")
    )
    status_dist["orden"] = status_dist["estado_pricing"].map(status_order_key)
    status_dist = status_dist.sort_values("orden").drop(columns="orden")

    st.bar_chart(status_dist.set_index("estado_pricing")[["casos"]])

    render_info_box(
        "Cómo interpretar este gráfico",
        "Aquí ya no se usa la clasificación original del dataset como eje principal, sino una clasificación operativa recalculada. "
        "Esto permite ver mejor cuántas pólizas están realmente en zona rentable, cuántas están ajustadas y cuántas destruyen margen."
    )

    st.dataframe(status_dist, use_container_width=True, hide_index=True)

with g2:
    st.markdown("### Margen total por estado de pricing")
    margin_by_status = (
        df_filtered.groupby("estado_pricing", dropna=False)
        .agg(
            casos=(member_id_col, "count") if member_id_col else ("estado_pricing", "count"),
            margen_total=("margen_estimado", "sum"),
            margen_medio=("margen_estimado", "mean"),
        )
        .reset_index()
    )
    margin_by_status["orden"] = margin_by_status["estado_pricing"].map(status_order_key)
    margin_by_status = margin_by_status.sort_values("orden").drop(columns="orden")

    st.bar_chart(margin_by_status.set_index("estado_pricing")[["margen_total"]])

    render_info_box(
        "Cómo interpretar este gráfico",
        "No muestra solo cuántos casos hay, sino cuánto dinero gana o pierde la cartera en cada estado. "
        "Es útil porque un grupo pequeño puede destruir mucho margen, mientras que otro grupo grande puede ser económicamente sano."
    )

    st.dataframe(
        margin_by_status.style.format({
            "margen_total": lambda x: fmt_num(x),
            "margen_medio": lambda x: fmt_num(x),
        }),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# BLOQUE ECONÓMICO
# =========================================================
st.markdown("### Lectura económica agregada")

e1, e2 = st.columns(2)

with e1:
    summary_totals = pd.DataFrame({
        "métrica": [
            "Prima total",
            "Coste total de referencia",
            "Margen neto total",
            "Ganancia total bruta",
            "Pérdida total bruta",
        ],
        "valor": [
            prima_total,
            coste_total,
            margen_total,
            ganancia_total,
            perdida_total,
        ]
    })
    st.dataframe(
        summary_totals.style.format({"valor": lambda x: fmt_num(x)}),
        use_container_width=True,
        hide_index=True
    )

with e2:
    gain_loss_chart = pd.DataFrame({
        "concepto": ["Ganancia bruta", "Pérdida bruta"],
        "importe": [ganancia_total, perdida_total]
    })
    st.bar_chart(gain_loss_chart.set_index("concepto")[["importe"]])

    render_info_box(
        "Cómo interpretar este bloque",
        "Aquí se separa claramente cuánto valor económico aporta la parte sana de la cartera y cuánto destruyen los casos deficitarios. "
        "Esto permite entender si el problema es absorbible o si exige rediseño de planes/primas."
    )


# =========================================================
# BOLSAS DE PROBLEMA POR SEGMENTO
# =========================================================
st.markdown("### Resumen por segmento o tipo de cliente")

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
        "Agrupar resumen por",
        options=[label for label, _ in group_candidates],
        index=0
    )
    selected_group_col = dict(group_candidates)[selected_group_label]

    segment_summary = (
        df_filtered.groupby(selected_group_col, dropna=False)
        .agg(
            clientes=(member_id_col, "count") if member_id_col else (selected_group_col, "count"),
            prima_media=("prima_actual", "mean"),
            coste_medio=("coste_referencia", "mean"),
            margen_total=("margen_estimado", "sum"),
            margen_medio=("margen_estimado", "mean"),
            ganancia_total=("ganancia_bruta", "sum"),
            perdida_total=("perdida_bruta", "sum"),
            pct_rentable=("rentable_flag", "mean"),
            pct_perdida_clara=("perdida_clara_flag", "mean"),
            ratio_medio=("ratio_prima_coste", "mean"),
        )
        .reset_index()
        .rename(columns={selected_group_col: "grupo"})
    )

    segment_summary["pct_rentable"] = segment_summary["pct_rentable"] * 100
    segment_summary["pct_perdida_clara"] = segment_summary["pct_perdida_clara"] * 100

    segment_summary = segment_summary.sort_values(
        ["perdida_total", "pct_perdida_clara", "margen_total"],
        ascending=[False, False, True]
    )

    render_info_box(
        "Cómo interpretar esta tabla",
        "Sirve para ver rápidamente en qué grupos se concentra la pérdida económica. "
        "En vez de revisar cliente por cliente desde el inicio, puedes entrar primero por el segmento que más pérdida total genera o que tiene mayor porcentaje de casos claramente deficitarios."
    )

    st.dataframe(
        segment_summary.style.format({
            "prima_media": lambda x: fmt_num(x),
            "coste_medio": lambda x: fmt_num(x),
            "margen_total": lambda x: fmt_num(x),
            "margen_medio": lambda x: fmt_num(x),
            "ganancia_total": lambda x: fmt_num(x),
            "perdida_total": lambda x: fmt_num(x),
            "pct_rentable": lambda x: fmt_pct(x),
            "pct_perdida_clara": lambda x: fmt_pct(x),
            "ratio_medio": lambda x: fmt_num(x),
        }),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No hay columnas suficientes para construir un resumen por segmentos.")


# =========================================================
# PANEL OPERATIVO CLIENTE A CLIENTE
# =========================================================
st.markdown("### Panel operativo cliente a cliente")

st.caption(
    "Primero revisa el resumen por segmento y luego baja al detalle individual. "
    "La tabla se ordena priorizando mayor pérdida, peor ratio y mayor riesgo cuando existe."
)

sort_cols = []
ascending = []

if "perdida_bruta" in df_filtered.columns:
    sort_cols.append("perdida_bruta")
    ascending.append(False)

if "ratio_prima_coste" in df_filtered.columns:
    sort_cols.append("ratio_prima_coste")
    ascending.append(True)

if risk_prob_col:
    sort_cols.append(risk_prob_col)
    ascending.append(False)

review_df = df_filtered.sort_values(sort_cols, ascending=ascending).copy() if sort_cols else df_filtered.copy()

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
    "ganancia_bruta",
    "perdida_bruta",
    "ratio_prima_coste",
    "estado_pricing",
    "recomendacion_pricing",
    "plan_sugerido_final",
    "justificacion_recomendacion",
]
panel_cols = [c for c in panel_cols if c in review_df.columns]

panel_rename = {
    member_id_col: "member_id" if member_id_col else None,
    policy_id_col: "policy_id" if policy_id_col else None,
    plan_type_col: "tipo_plan" if plan_type_col else None,
    coverage_scope_col: "cobertura" if coverage_scope_col else None,
    risk_segment_col: "segmento_riesgo" if risk_segment_col else None,
    risk_prob_col: "prob_riesgo" if risk_prob_col else None,
    claims_col: "claims" if claims_col else None,
}
panel_rename = {k: v for k, v in panel_rename.items() if k}

render_info_box(
    "Cómo usar este bloque",
    "Puedes buscar por póliza o cliente desde la barra lateral y revisar primero los casos con mayor pérdida bruta. "
    "La recomendación no sustituye la decisión de negocio, pero ayuda a priorizar revisión de prima, cobertura o migración de plan."
)

st.dataframe(
    review_df[panel_cols].rename(columns=panel_rename).head(100),
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
elif policy_id_col:
    selector_options = review_df[policy_id_col].fillna("Sin dato").astype(str).tolist()
elif member_id_col:
    selector_options = review_df[member_id_col].fillna("Sin dato").astype(str).tolist()

if selector_options:
    selected_case = st.selectbox(
        "Seleccionar cliente / póliza",
        options=selector_options,
        index=0
    )

    if member_id_col and policy_id_col:
        member_val, policy_val = [x.strip() for x in selected_case.split("|", 1)]
        case_df = review_df[
            (review_df[member_id_col].astype(str) == member_val) &
            (review_df[policy_id_col].astype(str) == policy_val)
        ]
    elif policy_id_col:
        case_df = review_df[review_df[policy_id_col].astype(str) == selected_case]
    else:
        case_df = review_df[review_df[member_id_col].astype(str) == selected_case]

    if not case_df.empty:
        row = case_df.iloc[0]

        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Prima actual", fmt_num(row.get("prima_actual", np.nan)))
        f2.metric("Coste de referencia", fmt_num(row.get("coste_referencia", np.nan)))
        f3.metric("Margen estimado", fmt_num(row.get("margen_estimado", np.nan)))
        f4.metric("Ratio prima / coste", fmt_num(row.get("ratio_prima_coste", np.nan)))

        with st.container(border=True):
            st.markdown("### Lectura individual")
            if member_id_col:
                st.markdown(f"- **member_id:** {row.get(member_id_col, 'Sin dato')}")
            if policy_id_col:
                st.markdown(f"- **policy_id:** {row.get(policy_id_col, 'Sin dato')}")
            if plan_type_col:
                st.markdown(f"- **Tipo de plan actual:** {row.get(plan_type_col, 'Sin dato')}")
            if coverage_scope_col:
                st.markdown(f"- **Cobertura:** {row.get(coverage_scope_col, 'Sin dato')}")
            if risk_segment_col:
                st.markdown(f"- **Segmento de riesgo:** {row.get(risk_segment_col, 'Sin dato')}")

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
    st.info("No hay identificadores suficientes para construir la ficha individual.")


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
    "ganancia_bruta",
    "perdida_bruta",
    "ratio_prima_coste",
    "estado_pricing",
    "estado_pricing_original",
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
