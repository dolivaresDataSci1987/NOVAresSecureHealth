import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_data import load_policy_member_master, load_provider_master


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Abuso de póliza y fraude",
    page_icon="🚨",
    layout="wide"
)


# =========================================================
# DATA
# =========================================================
@st.cache_data
def get_member_data():
    return load_policy_member_master()


@st.cache_data
def get_provider_data():
    return load_provider_master()


df = get_member_data().copy()
provider_df = get_provider_data().copy()

st.title("🚨 Abuso de póliza y fraude")
st.caption(
    "Detección de uso abusivo por cliente y revisión secundaria de proveedores con señales de fraude."
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


def build_score_buckets(series):
    s = to_numeric(series).dropna()
    if s.empty:
        return pd.DataFrame()

    bins = [-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf]
    labels = [
        "Muy bajo (<= 0,20)",
        "Bajo (0,20–0,40)",
        "Medio (0,40–0,60)",
        "Alto (0,60–0,80)",
        "Muy alto (> 0,80)",
    ]
    bucketed = pd.cut(s, bins=bins, labels=labels, include_lowest=True)
    out = (
        bucketed.value_counts(dropna=False)
        .reindex(labels, fill_value=0)
        .reset_index()
    )
    out.columns = ["tramo", "casos"]
    return out


def explain_abuse_reason(reason):
    if pd.isna(reason):
        return "Sin dato"

    reason_str = str(reason).strip()
    reason_key = reason_str.lower()

    abuse_reason_dict = {
        "high_claim_volume": "Volumen de reclamos anormalmente alto para el perfil del asegurado.",
        "high_base_cost": "Coste sanitario base elevado frente a lo esperado para casos comparables.",
        "flagged_provider_exposure": "Alta exposición a proveedores previamente marcados o sospechosos.",
        "high_flagged_provider_cost": "Parte importante del coste proviene de proveedores marcados.",
        "high_flagged_provider_use": "Uso frecuente de proveedores con señales de fraude o comportamiento anómalo.",
        "cost_frequency_mismatch": "La combinación entre frecuencia de uso y coste total no parece consistente con un patrón normal.",
        "high_cost_outlier": "Coste total claramente extremo respecto al resto de asegurados comparables.",
        "high_claim_frequency": "Frecuencia de reclamos superior a la esperada.",
        "repeat_flagged_provider_pattern": "Patrón repetido de utilización de proveedores marcados.",
        "network_abuse_pattern": "Uso de la red asistencial con patrón potencialmente oportunista o abusivo.",
        "provider_risk_exposure": "Elevada exposición a proveedores con mayor riesgo de fraude.",
        "mixed_abuse_pattern": "Combinación de varias señales de abuso, sin depender de una sola causa.",
        "high_utilization_pattern": "Uso intensivo de prestaciones o servicios por encima de lo habitual.",
        "unusual_claim_pattern": "Patrón de reclamos atípico frente al comportamiento esperado.",
        "chronic_overuse_pattern": "Uso excesivo recurrente no explicado solo por cronicidad.",
        "high_cost_and_frequency": "Coinciden alta frecuencia de uso y alto coste, elevando la sospecha de abuso.",
    }

    return abuse_reason_dict.get(
        reason_key,
        f"Señal técnica detectada: {reason_str.replace('_', ' ')}."
    )


def explain_provider_reason(reason):
    if pd.isna(reason):
        return "Sin dato"

    reason_str = str(reason).strip()
    reason_key = reason_str.lower()

    provider_reason_dict = {
        "high_claim_volume": "Volumen de reclamos elevado para ese proveedor.",
        "high_total_cost": "Coste total facturado anormalmente alto.",
        "high_cost_per_claim": "Coste medio por reclamo superior a lo esperado.",
        "billing_anomaly": "Posible anomalía en el patrón de facturación.",
        "repeat_member_pattern": "Patrón repetitivo con determinados pacientes o asegurados.",
        "suspicious_specialty_pattern": "Comportamiento atípico dentro de su especialidad.",
        "high_flag_rate": "Acumula múltiples señales de alerta en el modelo.",
        "mixed_fraud_pattern": "Combina varias señales de fraude o facturación anómala.",
    }

    return provider_reason_dict.get(
        reason_key,
        f"Señal técnica detectada: {reason_str.replace('_', ' ')}."
    )


def render_chart_explanation(title, text):
    with st.container(border=True):
        st.markdown(f"**Cómo interpretar este bloque · {title}**")
        st.markdown(text)


def build_dictionary_df(dictionary_map):
    return pd.DataFrame(
        [{"término": k, "explicación": v} for k, v in dictionary_map.items()]
    )


# =========================================================
# COLUMNAS DISPONIBLES
# =========================================================
member_id_col = "member_id" if "member_id" in df.columns else None
policy_id_col = "policy_id" if "policy_id" in df.columns else None
plan_type_col = "plan_type" if "plan_type" in df.columns else None
coverage_scope_col = "coverage_scope" if "coverage_scope" in df.columns else None

abuse_score_col = "member_abuse_score" if "member_abuse_score" in df.columns else None
abuse_severity_col = "member_abuse_severity" if "member_abuse_severity" in df.columns else None
abuse_reason_col = "member_abuse_reason" if "member_abuse_reason" in df.columns else None

claims_col = "claims_count" if "claims_count" in df.columns else ("claims_n" if "claims_n" in df.columns else None)
approved_cost_col = "approved_cost_sum" if "approved_cost_sum" in df.columns else None
risk_prob_col = "predicted_risk_probability" if "predicted_risk_probability" in df.columns else None
risk_segment_col = "predicted_risk_segment" if "predicted_risk_segment" in df.columns else None

flagged_provider_claims_n_col = "flagged_provider_claims_n" if "flagged_provider_claims_n" in df.columns else None
flagged_provider_claims_pct_col = "flagged_provider_claims_pct" if "flagged_provider_claims_pct" in df.columns else None
flagged_provider_cost_sum_col = "flagged_provider_cost_sum" if "flagged_provider_cost_sum" in df.columns else None
flagged_provider_cost_pct_col = "flagged_provider_cost_pct" if "flagged_provider_cost_pct" in df.columns else None
provider_fraud_score_max_col = "provider_fraud_score_max" if "provider_fraud_score_max" in df.columns else None

provider_id_col = "provider_id" if "provider_id" in provider_df.columns else None
provider_name_col = "provider_name" if "provider_name" in provider_df.columns else None
provider_type_col = "provider_type" if "provider_type" in provider_df.columns else None
provider_specialty_col = "specialty_group" if "specialty_group" in provider_df.columns else None
provider_score_col = "provider_fraud_score" if "provider_fraud_score" in provider_df.columns else None
provider_severity_col = "provider_fraud_severity" if "provider_fraud_severity" in provider_df.columns else None
provider_reason_col = "provider_fraud_reason" if "provider_fraud_reason" in provider_df.columns else None
provider_claims_col = "claims_count" if "claims_count" in provider_df.columns else ("claims_n_total" if "claims_n_total" in provider_df.columns else None)


# =========================================================
# NORMALIZACIÓN
# =========================================================
for col in [
    abuse_score_col,
    claims_col,
    approved_cost_col,
    risk_prob_col,
    flagged_provider_claims_n_col,
    flagged_provider_claims_pct_col,
    flagged_provider_cost_sum_col,
    flagged_provider_cost_pct_col,
    provider_fraud_score_max_col,
]:
    if col:
        df[col] = to_numeric(df[col])

for col in [provider_score_col, provider_claims_col]:
    if col:
        provider_df[col] = to_numeric(provider_df[col])

if abuse_severity_col:
    df[abuse_severity_col] = df[abuse_severity_col].apply(normalize_severity)

if risk_segment_col:
    df[risk_segment_col] = df[risk_segment_col].apply(normalize_severity)

if provider_severity_col:
    provider_df[provider_severity_col] = provider_df[provider_severity_col].apply(normalize_severity)


# =========================================================
# MÉTRICAS DERIVADAS
# =========================================================
if abuse_score_col:
    df["abuso_flag_operativo"] = df[abuse_score_col] >= 0.60
else:
    df["abuso_flag_operativo"] = False

if abuse_severity_col:
    df["abuso_alto_flag"] = df[abuse_severity_col].astype(str).isin(["Alto", "Muy alto"])
else:
    df["abuso_alto_flag"] = False

if approved_cost_col:
    total_cost = df[approved_cost_col].sum()
else:
    total_cost = np.nan

if flagged_provider_cost_sum_col and approved_cost_col:
    df["peso_coste_proveedor_marcado"] = np.where(
        df[approved_cost_col] > 0,
        df[flagged_provider_cost_sum_col] / df[approved_cost_col],
        np.nan
    )
else:
    df["peso_coste_proveedor_marcado"] = np.nan

if provider_score_col:
    provider_df["fraude_alto_flag"] = provider_df[provider_score_col] >= 0.60
else:
    provider_df["fraude_alto_flag"] = False

if provider_severity_col:
    provider_df["fraude_severidad_alta_flag"] = provider_df[provider_severity_col].astype(str).isin(["Alto", "Muy alto"])
else:
    provider_df["fraude_severidad_alta_flag"] = False


# =========================================================
# DICCIONARIOS
# =========================================================
member_metric_dict = {
    "member_abuse_score": "Score continuo de sospecha de abuso a nivel cliente. Cuanto más alto, mayor prioridad de revisión.",
    "member_abuse_severity": "Clasificación cualitativa del nivel de sospecha de abuso.",
    "member_abuse_reason": "Motivo principal que explica por qué el modelo marca el caso.",
    "claims_count": "Número de reclamos o usos registrados para el asegurado.",
    "approved_cost_sum": "Coste total aprobado asociado al asegurado.",
    "predicted_risk_probability": "Probabilidad estimada de riesgo clínico o asistencial del asegurado.",
    "predicted_risk_segment": "Segmento de riesgo estimado del asegurado.",
    "flagged_provider_claims_n": "Número de reclamos realizados con proveedores previamente marcados.",
    "flagged_provider_claims_pct": "Porcentaje de reclamos asociados a proveedores marcados.",
    "flagged_provider_cost_sum": "Coste total asociado a proveedores marcados.",
    "flagged_provider_cost_pct": "Porcentaje del coste total que procede de proveedores marcados.",
    "provider_fraud_score_max": "Máximo nivel de riesgo de fraude observado entre los proveedores vinculados al cliente.",
}

abuse_reason_dict_full = {
    "high_claim_volume": "Volumen de reclamos anormalmente alto para el perfil del asegurado.",
    "high_base_cost": "Coste sanitario base elevado frente a lo esperado para perfiles comparables.",
    "flagged_provider_exposure": "Alta exposición a proveedores ya marcados o sospechosos.",
    "high_flagged_provider_cost": "Parte importante del coste del cliente procede de proveedores marcados.",
    "high_flagged_provider_use": "Uso frecuente de proveedores con señales previas de comportamiento anómalo.",
    "cost_frequency_mismatch": "La combinación de frecuencia y coste no parece consistente con un uso normal.",
    "high_cost_outlier": "Coste total claramente extremo respecto al resto de clientes comparables.",
    "high_claim_frequency": "Frecuencia de reclamos superior a la esperada.",
    "repeat_flagged_provider_pattern": "Patrón repetido de uso de proveedores marcados.",
    "network_abuse_pattern": "Uso de la red asistencial con patrón potencialmente oportunista o abusivo.",
    "provider_risk_exposure": "Elevada exposición a proveedores con mayor riesgo de fraude.",
    "mixed_abuse_pattern": "Combinación de varias señales de abuso en el mismo caso.",
    "high_utilization_pattern": "Uso intensivo de servicios o prestaciones por encima de lo habitual.",
    "unusual_claim_pattern": "Patrón de reclamos atípico frente al comportamiento esperado.",
    "chronic_overuse_pattern": "Uso excesivo recurrente no explicado solo por cronicidad.",
    "high_cost_and_frequency": "Coinciden alta frecuencia de uso y alto coste.",
}

provider_reason_dict_full = {
    "high_claim_volume": "Volumen de reclamos elevado para ese proveedor.",
    "high_total_cost": "Coste total facturado anormalmente alto.",
    "high_cost_per_claim": "Coste medio por reclamo superior a lo esperado.",
    "billing_anomaly": "Posible anomalía en el patrón de facturación.",
    "repeat_member_pattern": "Patrón repetitivo con determinados asegurados.",
    "suspicious_specialty_pattern": "Comportamiento atípico dentro de su especialidad.",
    "high_flag_rate": "Acumula múltiples señales de alerta en el modelo.",
    "mixed_fraud_pattern": "Combina varias señales de fraude o facturación anómala.",
}


# =========================================================
# TABS
# =========================================================
tab1, tab2 = st.tabs(["Abuso por cliente", "Fraude por proveedor"])


# =========================================================
# TAB 1 - CLIENTES
# =========================================================
with tab1:
    st.markdown("## Identificación de clientes con posible uso abusivo")

    st.caption(
        "El objetivo no es detectar solo clientes caros, sino clientes cuyo patrón de uso, frecuencia de reclamos "
        "o exposición a proveedores marcados sugiera un posible abuso de la póliza."
    )

    with st.expander("Diccionario de métricas y señales de abuso", expanded=False):
        st.markdown("### Métricas principales")
        st.dataframe(
            build_dictionary_df(member_metric_dict),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Señales técnicas de abuso")
        st.dataframe(
            build_dictionary_df(abuse_reason_dict_full),
            use_container_width=True,
            hide_index=True
        )

    # -----------------------------------------------------
    # SIDEBAR / FILTROS
    # -----------------------------------------------------
    st.sidebar.header("Filtros · Abuso por cliente")

    df_filtered = df.copy()

    if abuse_severity_col:
        severity_options = sorted(
            df_filtered[abuse_severity_col].dropna().astype(str).unique().tolist(),
            key=severity_order_key
        )
        selected_severity = st.sidebar.multiselect(
            "Severidad de abuso",
            options=severity_options,
            default=severity_options
        )
        if selected_severity:
            df_filtered = df_filtered[df_filtered[abuse_severity_col].astype(str).isin(selected_severity)]

    if plan_type_col:
        plan_options = sorted(df_filtered[plan_type_col].dropna().astype(str).unique().tolist())
        selected_plans = st.sidebar.multiselect(
            "Tipo de plan",
            options=plan_options,
            default=plan_options
        )
        if selected_plans:
            df_filtered = df_filtered[df_filtered[plan_type_col].astype(str).isin(selected_plans)]

    only_high_abuse = st.sidebar.checkbox("Mostrar solo abuso alto / muy alto", value=False)
    if only_high_abuse:
        df_filtered = df_filtered[df_filtered["abuso_alto_flag"]]

    if abuse_score_col and df_filtered[abuse_score_col].notna().any():
        score_min = float(df_filtered[abuse_score_col].min())
        score_max = float(df_filtered[abuse_score_col].max())
        if score_min < score_max:
            selected_score = st.sidebar.slider(
                "Rango de abuse score",
                min_value=float(score_min),
                max_value=float(score_max),
                value=(float(score_min), float(score_max))
            )
            df_filtered = df_filtered[
                df_filtered[abuse_score_col].between(selected_score[0], selected_score[1], inclusive="both")
            ]

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

    # -----------------------------------------------------
    # KPIs
    # -----------------------------------------------------
    n_rows = len(df_filtered)

    abuso_score_medio = df_filtered[abuse_score_col].mean() if abuse_score_col else np.nan
    abuso_alto_n = int(df_filtered["abuso_alto_flag"].sum()) if n_rows > 0 else 0
    abuso_alto_pct = (abuso_alto_n / n_rows * 100) if n_rows > 0 else np.nan

    claims_totales = df_filtered[claims_col].sum() if claims_col else np.nan
    claims_medios = df_filtered[claims_col].mean() if claims_col else np.nan

    coste_total_filtrado = df_filtered[approved_cost_col].sum() if approved_cost_col else np.nan
    coste_medio_filtrado = df_filtered[approved_cost_col].mean() if approved_cost_col else np.nan

    coste_flagged_sum = df_filtered[flagged_provider_cost_sum_col].sum() if flagged_provider_cost_sum_col else np.nan
    pct_coste_flagged = (
        (coste_flagged_sum / coste_total_filtrado) * 100
        if approved_cost_col and pd.notna(coste_total_filtrado) and coste_total_filtrado > 0 and pd.notna(coste_flagged_sum)
        else np.nan
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Clientes analizados", fmt_int(n_rows))
    k2.metric("Abuso alto / muy alto", fmt_int(abuso_alto_n), delta=fmt_pct(abuso_alto_pct) if pd.notna(abuso_alto_pct) else None)
    k3.metric("Abuse score medio", fmt_num(abuso_score_medio))
    k4.metric("Claims medios", fmt_num(claims_medios))
    k5.metric("Coste total aprobado", fmt_num(coste_total_filtrado))
    k6.metric("Coste vinculado a proveedores marcados", fmt_pct(pct_coste_flagged) if pd.notna(pct_coste_flagged) else "N/A")

    # -----------------------------------------------------
    # LECTURA EJECUTIVA
    # -----------------------------------------------------
    insights = []

    if n_rows > 0:
        insights.append(f"La vista actual incluye **{fmt_int(n_rows)} clientes asegurados**.")

    if pd.notna(abuso_alto_pct):
        insights.append(
            f"El **{fmt_pct(abuso_alto_pct)}** presenta señales de **abuso alto o muy alto**, "
            "lo que permite acotar una bolsa prioritaria de revisión."
        )

    if pd.notna(abuso_score_medio):
        insights.append(
            f"El **abuse score medio** en la vista filtrada es **{fmt_num(abuso_score_medio)}**."
        )

    if pd.notna(pct_coste_flagged):
        insights.append(
            f"El **{fmt_pct(pct_coste_flagged)}** del coste aprobado visible está asociado a **proveedores ya marcados**, "
            "lo que puede reforzar la sospecha sobre determinados patrones de utilización."
        )

    if claims_col and pd.notna(claims_medios):
        insights.append(
            f"La frecuencia media de reclamos en esta vista es de **{fmt_num(claims_medios)}** por cliente."
        )

    with st.container(border=True):
        st.markdown("### Lectura de negocio")
        for item in insights:
            st.markdown(f"- {item}")

    # -----------------------------------------------------
    # DISTRIBUCIÓN
    # -----------------------------------------------------
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("### Distribución por severidad de abuso")
        if abuse_severity_col:
            sev_dist = (
                df_filtered[abuse_severity_col]
                .fillna("Sin dato")
                .value_counts()
                .rename_axis("severidad")
                .reset_index(name="casos")
            )
            sev_dist["orden"] = sev_dist["severidad"].map(severity_order_key)
            sev_dist = sev_dist.sort_values("orden").drop(columns="orden")
            st.bar_chart(sev_dist.set_index("severidad"))
            render_chart_explanation(
                "Distribución por severidad de abuso",
                "Este gráfico muestra cuántos clientes caen en cada nivel de severidad. "
                "Una concentración en **Alto** o **Muy alto** indica una cartera con mayor volumen de casos prioritarios para revisión. "
                "Si la mayor parte se concentra en niveles bajos o medios, el problema puede estar más focalizado en bolsillos concretos que en toda la cartera."
            )
            st.dataframe(sev_dist, use_container_width=True, hide_index=True)
        else:
            st.info("No existe la columna de severidad de abuso.")

    with d2:
        st.markdown("### Distribución del abuse score")
        if abuse_score_col and df_filtered[abuse_score_col].notna().any():
            score_dist = build_score_buckets(df_filtered[abuse_score_col])
            st.bar_chart(score_dist.set_index("tramo"))
            render_chart_explanation(
                "Distribución del abuse score",
                "El score resume la intensidad de sospecha de abuso. "
                "Si muchos casos se concentran en tramos **altos** o **muy altos**, aumenta la necesidad de revisión operativa. "
                "Si la distribución está más desplazada hacia valores bajos, la sospecha está menos extendida y más concentrada en casos puntuales."
            )
            st.dataframe(score_dist, use_container_width=True, hide_index=True)
        else:
            st.info("No existe información suficiente de abuse score.")

    # -----------------------------------------------------
    # RAZONES Y CONCENTRACIÓN
    # -----------------------------------------------------
    r1, r2 = st.columns(2)

    with r1:
        st.markdown("### Motivos de abuso más frecuentes")
        if abuse_reason_col:
            reason_dist = (
                df_filtered[abuse_reason_col]
                .fillna("Sin dato")
                .astype(str)
                .value_counts()
                .head(10)
                .rename_axis("motivo")
                .reset_index(name="casos")
            )
            reason_dist["explicación"] = reason_dist["motivo"].apply(explain_abuse_reason)
            st.bar_chart(reason_dist.set_index("motivo")[["casos"]])
            render_chart_explanation(
                "Motivos de abuso más frecuentes",
                "Aquí se observa qué señales técnicas aparecen con más frecuencia. "
                "No significa que todos los casos respondan a fraude real, sino que el modelo detecta patrones que merecen revisión. "
                "Una concentración en pocos motivos sugiere un patrón dominante; una distribución más dispersa indica casuística más heterogénea."
            )
            st.dataframe(reason_dist, use_container_width=True, hide_index=True)
        else:
            st.info("No existe la columna de motivo de abuso.")

    with r2:
        st.markdown("### Coste aprobado por severidad de abuso")
        if abuse_severity_col and approved_cost_col:
            cost_by_sev = (
                df_filtered
                .groupby(abuse_severity_col, dropna=False)
                .agg(
                    clientes=(member_id_col, "count") if member_id_col else (approved_cost_col, "count"),
                    coste_total=(approved_cost_col, "sum"),
                    coste_medio=(approved_cost_col, "mean")
                )
                .reset_index()
            )
            cost_by_sev["orden"] = cost_by_sev[abuse_severity_col].map(severity_order_key)
            cost_by_sev = cost_by_sev.sort_values("orden").drop(columns="orden")
            st.bar_chart(cost_by_sev.set_index(abuse_severity_col)[["coste_total"]])
            render_chart_explanation(
                "Coste aprobado por severidad de abuso",
                "Este bloque permite ver si el coste está concentrado en los casos de mayor severidad. "
                "Si los niveles **Altos** y **Muy altos** acumulan una gran parte del coste, la revisión prioritaria puede generar más impacto económico. "
                "Si el coste está más repartido, conviene revisar también patrones de frecuencia y no solo importes."
            )
            st.dataframe(
                cost_by_sev.rename(columns={abuse_severity_col: "severidad_abuso"}).style.format({
                    "coste_total": lambda x: fmt_num(x),
                    "coste_medio": lambda x: fmt_num(x),
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No hay columnas suficientes para mostrar coste por severidad.")

    # -----------------------------------------------------
    # BOLSAS DE ABUSO
    # -----------------------------------------------------
    st.markdown("### Bolsas de abuso por segmento de cartera")

    group_candidates = []
    if plan_type_col:
        group_candidates.append(("Tipo de plan", plan_type_col))
    if coverage_scope_col:
        group_candidates.append(("Cobertura", coverage_scope_col))
    if risk_segment_col:
        group_candidates.append(("Segmento de riesgo", risk_segment_col))
    if abuse_reason_col:
        group_candidates.append(("Motivo principal de abuso", abuse_reason_col))

    if group_candidates:
        selected_group_label = st.selectbox(
            "Agrupar análisis por",
            options=[label for label, _ in group_candidates],
            index=0
        )
        selected_group_col = dict(group_candidates)[selected_group_label]

        agg_dict = {
            member_id_col if member_id_col else policy_id_col: "count",
        }
        if abuse_score_col:
            agg_dict[abuse_score_col] = "mean"
        if claims_col:
            agg_dict[claims_col] = "mean"
        if approved_cost_col:
            agg_dict[approved_cost_col] = "mean"
        if flagged_provider_cost_pct_col:
            agg_dict[flagged_provider_cost_pct_col] = "mean"
        agg_dict["abuso_alto_flag"] = "mean"

        group_summary = (
            df_filtered
            .groupby(selected_group_col, dropna=False)
            .agg(agg_dict)
            .reset_index()
        )

        group_summary = group_summary.rename(columns={
            selected_group_col: "grupo",
            member_id_col if member_id_col else policy_id_col: "clientes",
            abuse_score_col: "abuse_score_medio" if abuse_score_col else "abuse_score_medio",
            claims_col: "claims_medios" if claims_col else "claims_medios",
            approved_cost_col: "coste_medio" if approved_cost_col else "coste_medio",
            flagged_provider_cost_pct_col: "pct_coste_proveedor_marcado" if flagged_provider_cost_pct_col else "pct_coste_proveedor_marcado",
            "abuso_alto_flag": "pct_abuso_alto",
        })

        group_summary["pct_abuso_alto"] = group_summary["pct_abuso_alto"] * 100
        group_summary = group_summary.sort_values(["pct_abuso_alto"], ascending=False)

        style_map = {
            "pct_abuso_alto": lambda x: fmt_pct(x),
        }
        for col in ["abuse_score_medio", "claims_medios", "coste_medio", "pct_coste_proveedor_marcado"]:
            if col in group_summary.columns:
                if "pct_" in col:
                    style_map[col] = lambda x: fmt_pct(x)
                else:
                    style_map[col] = lambda x: fmt_num(x)

        render_chart_explanation(
            "Bolsas de abuso por segmento de cartera",
            "Esta tabla sirve para detectar en qué segmentos se concentra más la sospecha. "
            "Los grupos con mayor **porcentaje de abuso alto**, mayor **score medio** o mayor **coste medio** deberían revisarse antes. "
            "Es útil para pasar de una lógica de caso individual a una lógica de gestión por segmentos."
        )

        st.dataframe(
            group_summary.style.format(style_map),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No hay columnas de agrupación suficientes.")

    # -----------------------------------------------------
    # CASOS PRIORITARIOS
    # -----------------------------------------------------
    st.markdown("### Casos prioritarios para revisión manual")

    sort_cols = []
    ascending = []

    if abuse_score_col:
        sort_cols.append(abuse_score_col)
        ascending.append(False)
    if flagged_provider_cost_sum_col:
        sort_cols.append(flagged_provider_cost_sum_col)
        ascending.append(False)
    if approved_cost_col:
        sort_cols.append(approved_cost_col)
        ascending.append(False)

    if sort_cols:
        review_df = df_filtered.sort_values(sort_cols, ascending=ascending).copy()
    else:
        review_df = df_filtered.copy()

    review_cols = [
        member_id_col,
        policy_id_col,
        plan_type_col,
        risk_segment_col,
        risk_prob_col,
        abuse_score_col,
        abuse_severity_col,
        abuse_reason_col,
        claims_col,
        approved_cost_col,
        flagged_provider_claims_n_col,
        flagged_provider_claims_pct_col,
        flagged_provider_cost_sum_col,
        flagged_provider_cost_pct_col,
        provider_fraud_score_max_col,
    ]
    review_cols = [c for c in review_cols if c and c in review_df.columns]

    if abuse_reason_col in review_df.columns:
        review_df["explicación_motivo_abuso"] = review_df[abuse_reason_col].apply(explain_abuse_reason)
        review_cols_with_text = review_cols + ["explicación_motivo_abuso"]
    else:
        review_cols_with_text = review_cols

    st.caption(
        "Se priorizan clientes con mayor abuse score, mayor coste y mayor exposición a proveedores marcados."
    )
    render_chart_explanation(
        "Casos prioritarios para revisión manual",
        "Esta tabla no confirma fraude ni abuso real por sí sola. "
        "Sirve para priorizar expedientes que combinan varias señales: intensidad del score, coste elevado y exposición a proveedores marcados. "
        "Es el punto de entrada para auditoría manual o revisión clínica/operativa."
    )
    st.dataframe(
        review_df[review_cols_with_text].head(25),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # VISTA DETALLADA
    # -----------------------------------------------------
    st.markdown("### Vista detallada de clientes")
    st.dataframe(
        review_df[review_cols_with_text],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# TAB 2 - PROVEEDORES
# =========================================================
with tab2:
    st.markdown("## Proveedores con señales de fraude")
    st.caption(
        "Vista secundaria para revisar proveedores anómalos que pueden estar amplificando el abuso observado en clientes."
    )

    with st.expander("Diccionario de señales de fraude por proveedor", expanded=False):
        st.dataframe(
            build_dictionary_df(provider_reason_dict_full),
            use_container_width=True,
            hide_index=True
        )

    n_providers = len(provider_df)
    flagged_providers = int(provider_df["fraude_severidad_alta_flag"].sum()) if "fraude_severidad_alta_flag" in provider_df.columns else 0
    flagged_pct = (flagged_providers / n_providers * 100) if n_providers > 0 else np.nan
    provider_score_mean = provider_df[provider_score_col].mean() if provider_score_col else np.nan

    p1, p2, p3 = st.columns(3)
    p1.metric("Proveedores analizados", fmt_int(n_providers))
    p2.metric("Fraude alto / muy alto", fmt_int(flagged_providers), delta=fmt_pct(flagged_pct) if pd.notna(flagged_pct) else None)
    p3.metric("Fraud score medio", fmt_num(provider_score_mean))

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("### Distribución por severidad")
        if provider_severity_col:
            sev_counts = (
                provider_df[provider_severity_col]
                .fillna("Sin dato")
                .value_counts()
                .rename_axis("severidad")
                .reset_index(name="casos")
            )
            sev_counts["orden"] = sev_counts["severidad"].map(severity_order_key)
            sev_counts = sev_counts.sort_values("orden").drop(columns="orden")
            st.bar_chart(sev_counts.set_index("severidad"))
            render_chart_explanation(
                "Distribución por severidad de fraude en proveedores",
                "Permite ver si el riesgo está repartido entre muchos proveedores o concentrado en unos pocos niveles altos. "
                "Una presencia relevante de proveedores en **Alto** o **Muy alto** puede indicar focos prioritarios de auditoría de red asistencial."
            )
            st.dataframe(sev_counts, use_container_width=True, hide_index=True)
        else:
            st.info("No existe la columna de severidad de fraude.")

    with d2:
        st.markdown("### Motivos de fraude más frecuentes")
        if provider_reason_col:
            reason_counts = (
                provider_df[provider_reason_col]
                .fillna("Sin dato")
                .astype(str)
                .value_counts()
                .head(10)
                .rename_axis("motivo")
                .reset_index(name="casos")
            )
            reason_counts["explicación"] = reason_counts["motivo"].apply(explain_provider_reason)
            st.bar_chart(reason_counts.set_index("motivo")[["casos"]])
            render_chart_explanation(
                "Motivos de fraude más frecuentes en proveedores",
                "Este bloque muestra qué patrones anómalos aparecen con mayor frecuencia entre los proveedores. "
                "Sirve para entender si el problema parece estar más relacionado con volumen, coste, facturación o recurrencia de comportamiento."
            )
            st.dataframe(reason_counts, use_container_width=True, hide_index=True)
        else:
            st.info("No existe la columna de motivo de fraude.")

    provider_cols = [
        provider_id_col,
        provider_name_col,
        provider_type_col,
        provider_specialty_col,
        provider_claims_col,
        provider_score_col,
        provider_severity_col,
        provider_reason_col,
    ]
    provider_cols = [c for c in provider_cols if c and c in provider_df.columns]

    if provider_score_col:
        provider_view = provider_df.sort_values(provider_score_col, ascending=False).copy()
    else:
        provider_view = provider_df.copy()

    if provider_reason_col in provider_view.columns:
        provider_view["explicación_motivo_fraude"] = provider_view[provider_reason_col].apply(explain_provider_reason)
        provider_cols_with_text = provider_cols + ["explicación_motivo_fraude"]
    else:
        provider_cols_with_text = provider_cols

    st.markdown("### Proveedores prioritarios")
    render_chart_explanation(
        "Proveedores prioritarios",
        "Esta tabla ordena los proveedores por nivel de sospecha. "
        "Los primeros registros deberían revisarse antes, especialmente si además acumulan volumen de reclamos o aparecen ligados a múltiples clientes marcados."
    )
    st.dataframe(
        provider_view[provider_cols_with_text].head(25),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Vista detallada de proveedores")
    st.dataframe(
        provider_view[provider_cols_with_text],
        use_container_width=True,
        hide_index=True
    )
