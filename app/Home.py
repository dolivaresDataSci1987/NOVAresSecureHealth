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

# =========================================================
# CARGA DE DATOS
# =========================================================
@st.cache_data
def get_policy_member():
    return load_policy_member_master()


@st.cache_data
def get_provider():
    return load_provider_master()


@st.cache_data
def get_prospect():
    return load_prospect_master()


# =========================================================
# HELPERS
# =========================================================
def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def safe_mean(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    s = safe_numeric(df[col])
    if s.dropna().empty:
        return None
    return float(s.mean())


def safe_sum(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    s = safe_numeric(df[col])
    if s.dropna().empty:
        return None
    return float(s.sum())


def safe_count_unique(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    return int(df[col].dropna().astype(str).nunique())


def normalize_text(series: pd.Series):
    return series.astype(str).str.strip().str.lower()


def fmt_int(x):
    if x is None or pd.isna(x):
        return "N/D"
    return f"{int(round(x)):,}".replace(",", ".")


def fmt_num(x, decimals=2):
    if x is None or pd.isna(x):
        return "N/D"
    return f"{x:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(x, decimals=1):
    if x is None or pd.isna(x):
        return "N/D"
    return f"{x:.{decimals}f}%"


def get_pct(part, total):
    if part is None or total in [None, 0]:
        return None
    return 100 * part / total


def value_count_df(df: pd.DataFrame, col: str, name_col="categoria", top_n=None):
    if col not in df.columns:
        return pd.DataFrame()
    out = (
        df[col]
        .fillna("Sin dato")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis(name_col)
        .reset_index(name="conteo")
    )
    if top_n is not None:
        out = out.head(top_n)
    return out


def mean_by_group(df: pd.DataFrame, group_col: str, value_col: str, top_n=None):
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    temp = df[[group_col, value_col]].copy()
    temp[value_col] = safe_numeric(temp[value_col])
    temp = temp.dropna(subset=[value_col])
    if temp.empty:
        return pd.DataFrame()
    out = (
        temp.groupby(group_col, dropna=False)[value_col]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    out[group_col] = out[group_col].fillna("Sin dato").astype(str)
    if top_n is not None:
        out = out.head(top_n)
    return out


def sum_by_group(df: pd.DataFrame, group_col: str, value_col: str, top_n=None):
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    temp = df[[group_col, value_col]].copy()
    temp[value_col] = safe_numeric(temp[value_col])
    temp = temp.dropna(subset=[value_col])
    if temp.empty:
        return pd.DataFrame()
    out = (
        temp.groupby(group_col, dropna=False)[value_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    out[group_col] = out[group_col].fillna("Sin dato").astype(str)
    if top_n is not None:
        out = out.head(top_n)
    return out


def high_risk_mask(df: pd.DataFrame):
    if "predicted_risk_segment" not in df.columns:
        return pd.Series(False, index=df.index)
    s = normalize_text(df["predicted_risk_segment"])
    return s.isin(["high", "very_high", "alto", "muy_alto", "muy alto"])


def high_abuse_mask(df: pd.DataFrame):
    if "member_abuse_severity" not in df.columns:
        return pd.Series(False, index=df.index)
    s = normalize_text(df["member_abuse_severity"])
    return s.isin(["high", "very_high", "alto", "muy_alto", "muy alto"])


def pricing_tension_mask(df: pd.DataFrame):
    if "pricing_adequacy_ratio" in df.columns:
        ratio = safe_numeric(df["pricing_adequacy_ratio"])
        return ratio.notna() & ((ratio < 0.90) | (ratio > 1.10))

    if "pricing_status" in df.columns:
        s = normalize_text(df["pricing_status"])
        return s.isin(["underpriced", "overpriced", "infra_precificado", "sobre_precificado"])

    return pd.Series(False, index=df.index)


def fraud_exposed_mask(df: pd.DataFrame):
    masks = []

    if "flagged_provider_claims_n" in df.columns:
        masks.append(safe_numeric(df["flagged_provider_claims_n"]).fillna(0) > 0)

    if "flagged_provider_cost_sum" in df.columns:
        masks.append(safe_numeric(df["flagged_provider_cost_sum"]).fillna(0) > 0)

    if "flagged_provider_claims_pct" in df.columns:
        masks.append(safe_numeric(df["flagged_provider_claims_pct"]).fillna(0) > 0)

    if not masks:
        return pd.Series(False, index=df.index)

    out = masks[0].copy()
    for m in masks[1:]:
        out = out | m
    return out


def build_priority_table(df: pd.DataFrame):
    temp = df.copy()

    temp["flag_riesgo_alto"] = high_risk_mask(temp).astype(int)
    temp["flag_tension_pricing"] = pricing_tension_mask(temp).astype(int)
    temp["flag_abuso_alto"] = high_abuse_mask(temp).astype(int)
    temp["flag_fraude_expuesto"] = fraud_exposed_mask(temp).astype(int)

    if "cancellation_flag" in temp.columns:
        temp["flag_cancelacion"] = safe_numeric(temp["cancellation_flag"]).fillna(0).astype(int)
    else:
        temp["flag_cancelacion"] = 0

    temp["prioridad_total"] = (
        temp["flag_riesgo_alto"]
        + temp["flag_tension_pricing"]
        + temp["flag_abuso_alto"]
        + temp["flag_fraude_expuesto"]
        + temp["flag_cancelacion"]
    )

    if "approved_cost_sum" in temp.columns:
        temp["approved_cost_sum"] = safe_numeric(temp["approved_cost_sum"])

    sort_cols = ["prioridad_total"]
    ascending = [False]

    if "approved_cost_sum" in temp.columns:
        sort_cols.append("approved_cost_sum")
        ascending.append(False)

    cols = [
        "member_id",
        "policy_id",
        "region",
        "plan_type",
        "plan_tier",
        "coverage_scope",
        "predicted_risk_segment",
        "pricing_adequacy_ratio",
        "member_abuse_severity",
        "flagged_provider_claims_n",
        "flagged_provider_cost_sum",
        "approved_cost_sum",
        "premium_monthly",
        "claims_count",
        "cancellation_flag",
        "prioridad_total",
    ]
    cols = [c for c in cols if c in temp.columns]

    return temp.sort_values(sort_cols, ascending=ascending)[cols].head(20).copy()


# =========================================================
# SAFE LOADS
# =========================================================
try:
    df = get_policy_member().copy()
except Exception as e:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.error(f"No se pudo cargar el dataset maestro principal: {e}")
    st.info("Revisa que el archivo de datos esté correctamente incluido en el proyecto y que la ruta de carga sea válida.")
    st.stop()

try:
    provider_df = get_provider().copy()
except Exception:
    provider_df = pd.DataFrame()

try:
    prospect_df = get_prospect().copy()
except Exception:
    prospect_df = pd.DataFrame()


# =========================================================
# HEADER
# =========================================================
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

st.markdown(
    """
    **Visión general ejecutiva del portfolio asegurado**  
    Esta vista resume tamaño de cartera, exposición al riesgo, adecuación de pricing,
    señales de abuso / fraude y composición general del negocio.

    **Aviso importante:** este dashboard es una herramienta analítica de apoyo a la decisión.
    No sustituye la evaluación actuarial, clínica, antifraude, de suscripción o de negocio.
    Algunas primas, costes o indicadores simulados pueden diferir de condiciones operativas reales.
    """
)

overview = dataset_overview(df)
n_rows = overview.get("rows", len(df))
n_cols = overview.get("columns", df.shape[1])

# =========================================================
# RESUMEN GENERAL
# =========================================================
st.markdown("## Resumen general")

n_members = safe_count_unique(df, "member_id")
n_policies = safe_count_unique(df, "policy_id")
n_providers = len(provider_df) if not provider_df.empty else None
n_prospects = len(prospect_df) if not prospect_df.empty else None
avg_premium = safe_mean(df, "premium_monthly")
total_approved_cost = safe_sum(df, "approved_cost_sum")
avg_loss_ratio = safe_mean(df, "observed_loss_ratio")

r1c1, r1c2, r1c3, r1c4 = st.columns(4)
r1c1.metric("Miembros analizados", fmt_int(n_members if n_members is not None else n_rows))
r1c2.metric("Pólizas analizadas", fmt_int(n_policies if n_policies is not None else n_rows))
r1c3.metric("Prestadores", fmt_int(n_providers))
r1c4.metric("Prospectos", fmt_int(n_prospects))

r2c1, r2c2, r2c3 = st.columns(3)
r2c1.metric("Prima mensual media", fmt_num(avg_premium))
r2c2.metric("Coste aprobado total", fmt_num(total_approved_cost))
r2c3.metric(
    "Siniestralidad observada media",
    fmt_pct(avg_loss_ratio * 100 if avg_loss_ratio is not None and avg_loss_ratio <= 2 else avg_loss_ratio)
)

# =========================================================
# SEÑALES CLAVE
# =========================================================
st.markdown("## Señales clave")

total_cases = len(df)

mask_risk = high_risk_mask(df)
mask_pricing = pricing_tension_mask(df)
mask_abuse = high_abuse_mask(df)
mask_fraud = fraud_exposed_mask(df)

high_risk_n = int(mask_risk.sum())
pricing_tension_n = int(mask_pricing.sum())
fraud_exposed_n = int(mask_fraud.sum())

s1, s2, s3 = st.columns(3)
s1.metric(
    "Riesgo alto / muy alto",
    fmt_int(high_risk_n),
    fmt_pct(get_pct(high_risk_n, total_cases))
)
s2.metric(
    "Tensión de pricing",
    fmt_int(pricing_tension_n),
    fmt_pct(get_pct(pricing_tension_n, total_cases))
)
s3.metric(
    "Exposición a fraude / abuso",
    fmt_int(fraud_exposed_n),
    fmt_pct(get_pct(fraud_exposed_n, total_cases))
)

# =========================================================
# LECTURA EJECUTIVA
# =========================================================
st.markdown("## Lectura ejecutiva")

e1, e2, e3 = st.columns(3)

with e1:
    st.markdown("### Riesgo")

    risk_cost = safe_sum(df.loc[mask_risk], "approved_cost_sum")
    risk_loss_ratio = safe_mean(df.loc[mask_risk], "observed_loss_ratio")

    st.write(f"**Casos de riesgo alto / muy alto:** {fmt_int(high_risk_n)}")
    st.write(f"**% del portfolio:** {fmt_pct(get_pct(high_risk_n, total_cases))}")
    st.write(f"**Coste aprobado del grupo:** {fmt_num(risk_cost)}")
    st.write(
        f"**Siniestralidad media del grupo:** "
        f"{fmt_pct(risk_loss_ratio * 100 if risk_loss_ratio is not None and risk_loss_ratio <= 2 else risk_loss_ratio)}"
    )

with e2:
    st.markdown("### Pricing")

    avg_suggested_mid = safe_mean(df, "suggested_premium_mid")
    avg_adequacy_ratio = safe_mean(df, "pricing_adequacy_ratio")
    sum_gap_expected = safe_sum(df, "premium_gap_vs_expected")
    tension_gap_expected = safe_sum(df.loc[mask_pricing], "premium_gap_vs_expected")

    st.write(f"**Prima mensual media actual:** {fmt_num(avg_premium)}")
    st.write(f"**Prima sugerida media:** {fmt_num(avg_suggested_mid)}")
    st.write(f"**Ratio medio de adecuación:** {fmt_num(avg_adequacy_ratio)}")
    st.write(f"**Brecha agregada vs esperado:** {fmt_num(sum_gap_expected)}")
    st.write(f"**Brecha en casos con tensión:** {fmt_num(tension_gap_expected)}")

with e3:
    st.markdown("### Abuso / fraude")

    high_abuse_n = int(mask_abuse.sum())
    fraud_cost = safe_sum(df.loc[mask_fraud], "flagged_provider_cost_sum")
    fraud_cost_pct = safe_mean(df.loc[mask_fraud], "flagged_provider_cost_pct")
    fraud_claims_n = safe_sum(df.loc[mask_fraud], "flagged_provider_claims_n")

    st.write(f"**Casos con abuso alto / muy alto:** {fmt_int(high_abuse_n)}")
    st.write(f"**Casos expuestos a prestadores marcados:** {fmt_int(fraud_exposed_n)}")
    st.write(f"**Claims ligados a prestadores marcados:** {fmt_int(fraud_claims_n)}")
    st.write(f"**Coste asociado a prestadores marcados:** {fmt_num(fraud_cost)}")
    st.write(f"**% medio de coste expuesto:** {fmt_pct(fraud_cost_pct)}")

# =========================================================
# EXPLORACIÓN GENERAL
# =========================================================
st.markdown("## Exploración general")

g1, g2 = st.columns(2)

with g1:
    st.markdown("#### Distribución por segmento de riesgo")
    risk_dist = value_count_df(df, "predicted_risk_segment", "segmento")
    if not risk_dist.empty:
        st.bar_chart(risk_dist.set_index("segmento"))
    else:
        st.info("No hay datos suficientes para mostrar este gráfico.")

with g2:
    st.markdown("#### Mix de cartera por plan")
    group_col_plan = "plan_tier" if "plan_tier" in df.columns else ("plan_type" if "plan_type" in df.columns else None)
    if group_col_plan:
        plan_dist = value_count_df(df, group_col_plan, "plan")
        if not plan_dist.empty:
            st.bar_chart(plan_dist.set_index("plan"))
        else:
            st.info("No hay datos suficientes para mostrar este gráfico.")
    else:
        st.info("No hay columnas de plan disponibles.")

g3, g4 = st.columns(2)

with g3:
    st.markdown("#### Distribución por cobertura")
    coverage_dist = value_count_df(df, "coverage_scope", "cobertura")
    if not coverage_dist.empty:
        st.bar_chart(coverage_dist.set_index("cobertura"))
    else:
        st.info("No hay datos suficientes para mostrar este gráfico.")

with g4:
    st.markdown("#### Prima mensual media por tipo de plan")
    econ_group_col = "plan_type" if "plan_type" in df.columns else ("plan_tier" if "plan_tier" in df.columns else None)
    if econ_group_col:
        premium_by_plan = mean_by_group(df, econ_group_col, "premium_monthly")
        if not premium_by_plan.empty:
            st.bar_chart(premium_by_plan.set_index(econ_group_col))
        else:
            st.info("No hay datos suficientes para mostrar este gráfico.")
    else:
        st.info("No hay columnas de plan disponibles para el análisis económico.")

# =========================================================
# BLOQUE ECONÓMICO ADICIONAL
# =========================================================
st.markdown("## Lectura económica")

eco1, eco2 = st.columns(2)

with eco1:
    st.markdown("#### Coste aprobado total por cobertura")
    cost_by_cov = sum_by_group(df, "coverage_scope", "approved_cost_sum")
    if not cost_by_cov.empty:
        st.bar_chart(cost_by_cov.set_index("coverage_scope"))
    else:
        st.info("No hay datos suficientes para mostrar este gráfico.")

with eco2:
    st.markdown("#### Prima anual media por cobertura")
    annual_premium_by_cov = mean_by_group(df, "coverage_scope", "premium_annual")
    if not annual_premium_by_cov.empty:
        st.bar_chart(annual_premium_by_cov.set_index("coverage_scope"))
    else:
        st.info("No hay datos suficientes para mostrar este gráfico.")

# =========================================================
# CASOS PRIORITARIOS
# =========================================================
st.markdown("## Casos prioritarios")
st.caption(
    "Selección orientativa de registros con mayor criticidad combinada entre riesgo, pricing, abuso, exposición a red marcada y cancelación."
)

priority_table = build_priority_table(df)

if priority_table.empty:
    st.info("No hay datos suficientes para construir la tabla de casos prioritarios.")
else:
    st.dataframe(priority_table, use_container_width=True, hide_index=True)

# =========================================================
# GLOSARIO DE TÉRMINOS
# =========================================================
st.markdown("## Glosario de términos")

with st.expander("Abrir glosario y guía rápida de interpretación", expanded=False):
    st.markdown(
        """
        Este glosario resume los conceptos principales utilizados en la Home para facilitar
        una lectura ejecutiva y homogénea del dashboard.
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            ### Cartera y estructura

            **Miembro analizado:** persona asegurada incluida en el universo de análisis del dashboard.

            **Póliza:** contrato o unidad de aseguramiento asociada al miembro y a sus coberturas.

            **Plan / tipo de plan:** categoría comercial o técnica del producto asegurador contratado.

            **Tier de plan:** nivel del plan dentro de una estructura escalonada, por ejemplo básico, intermedio o premium.

            **Cobertura:** alcance de protección o paquete asistencial incluido en la póliza.

            **Prestador:** proveedor sanitario, clínica, hospital, laboratorio o profesional que presta servicios cubiertos.

            **Prospecto:** potencial asegurado aún no incorporado definitivamente a cartera.
            """
        )

        st.markdown(
            """
            ### Economía y pricing

            **Prima mensual:** importe mensual cobrado o estimado para la póliza.

            **Prima anual:** equivalente anual de la prima del asegurado.

            **Prima sugerida:** prima estimada por el modelo como más coherente con el perfil de riesgo y coste esperado.

            **Brecha vs esperado:** diferencia entre la prima observada y la referencia esperada o técnica.

            **Ratio de adecuación tarifaria:** indicador que compara la prima observada con la prima esperada.
            Valores cercanos a 1 indican mejor alineación.

            **Tensión de pricing:** situación en la que la prima actual parece desviarse de forma relevante respecto al nivel esperado.

            **Coste aprobado:** gasto asistencial finalmente validado o aprobado para el caso o conjunto de claims.

            **Siniestralidad observada:** relación entre coste y prima, utilizada para valorar sostenibilidad técnica de la cartera.
            """
        )

    with c2:
        st.markdown(
            """
            ### Riesgo

            **Probabilidad de riesgo:** estimación del modelo sobre la probabilidad relativa de que un miembro genere mayor coste o complejidad.

            **Segmento de riesgo:** clasificación del miembro en niveles de riesgo, por ejemplo muy bajo, medio o alto.

            **Riesgo alto / muy alto:** subconjunto de miembros con mayor exposición esperada y, por tanto, mayor prioridad analítica.

            **Caso prioritario:** registro que combina varias señales relevantes, como alto riesgo, tensión de pricing,
            abuso elevado, exposición a prestadores marcados o cancelación.
            """
        )

        st.markdown(
            """
            ### Abuso, fraude y operación

            **Abuso:** utilización anómala o ineficiente del sistema que no necesariamente implica fraude intencional,
            pero sí puede generar sobrecoste.

            **Fraude:** comportamiento potencialmente irregular o deliberado orientado a obtener un beneficio indebido.

            **Prestador marcado:** proveedor con señales analíticas de comportamiento anómalo o patrón potencialmente irregular.

            **Claims:** eventos, actos asistenciales o solicitudes de reembolso / atención registrados en el sistema.

            **Claims ligados a prestadores marcados:** volumen de actividad asistencial relacionado con proveedores señalados.

            **Coste expuesto:** coste asociado a miembros o claims con señales de relación con red marcada o patrón sospechoso.

            **Exposición a fraude / abuso:** presencia de indicadores que justifican seguimiento reforzado desde la óptica antifraude o de uso indebido.

            **Cancelación:** indicador de baja, salida o interrupción de la continuidad de la póliza o relación aseguradora.

            **Portfolio:** conjunto agregado de miembros, pólizas, costes, primas y señales analíticas evaluadas en el dashboard.
            """
        )

# =========================================================
# CONTEXTO DEL DATASET
# =========================================================
st.markdown("## Contexto del dataset")

f1, f2 = st.columns(2)
f1.metric("Filas del maestro", fmt_int(n_rows))
f2.metric("Columnas del maestro", fmt_int(n_cols))
