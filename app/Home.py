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
def safe_numeric_mean(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.dropna().empty:
        return None
    return float(s.mean())


def safe_numeric_sum(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.dropna().empty:
        return None
    return float(s.sum())


def safe_count_unique(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    return int(df[col].dropna().astype(str).nunique())


def safe_value_count_df(df: pd.DataFrame, col: str, label_name: str = "categoria"):
    if col not in df.columns:
        return pd.DataFrame()
    out = (
        df[col]
        .fillna("Sin dato")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis(label_name)
        .reset_index(name="conteo")
    )
    return out


def pct(part, total):
    if total in [0, None]:
        return None
    return 100.0 * part / total


def fmt_int(x):
    if x is None or pd.isna(x):
        return "N/D"
    return f"{int(round(x)):,}".replace(",", ".")


def fmt_float(x, decimals=2):
    if x is None or pd.isna(x):
        return "N/D"
    return f"{x:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(x, decimals=1):
    if x is None or pd.isna(x):
        return "N/D"
    return f"{x:.{decimals}f}%"


def normalize_text_series(s: pd.Series):
    return s.astype(str).str.strip().str.lower()


def detect_high_risk_count(df: pd.DataFrame):
    if "predicted_risk_segment" not in df.columns:
        return None
    s = normalize_text_series(df["predicted_risk_segment"])
    return int(s.isin(["high", "very_high", "alto", "muy_alto", "muy alto"]).sum())


def detect_underpriced_count(df: pd.DataFrame):
    if "pricing_status" not in df.columns:
        return None
    s = normalize_text_series(df["pricing_status"])
    return int(s.isin(["underpriced", "infra-priced", "infra_precificado", "infra precificado"]).sum())


def detect_high_abuse_count(df: pd.DataFrame):
    if "member_abuse_severity" not in df.columns:
        return None
    s = normalize_text_series(df["member_abuse_severity"])
    return int(s.isin(["high", "very_high", "alto", "muy_alto", "muy alto"]).sum())


def build_priority_table(df: pd.DataFrame):
    temp = df.copy()

    if "predicted_risk_segment" in temp.columns:
        s = normalize_text_series(temp["predicted_risk_segment"])
        temp["flag_riesgo_alto"] = s.isin(["high", "very_high", "alto", "muy_alto", "muy alto"]).astype(int)
    else:
        temp["flag_riesgo_alto"] = 0

    if "pricing_status" in temp.columns:
        s = normalize_text_series(temp["pricing_status"])
        temp["flag_pricing_critico"] = s.isin(["underpriced", "infra-priced", "infra_precificado", "infra precificado"]).astype(int)
    else:
        temp["flag_pricing_critico"] = 0

    if "member_abuse_severity" in temp.columns:
        s = normalize_text_series(temp["member_abuse_severity"])
        temp["flag_abuso_alto"] = s.isin(["high", "very_high", "alto", "muy_alto", "muy alto"]).astype(int)
    else:
        temp["flag_abuso_alto"] = 0

    if "cancellation_flag" in temp.columns:
        temp["flag_cancelacion"] = pd.to_numeric(temp["cancellation_flag"], errors="coerce").fillna(0).astype(int)
    else:
        temp["flag_cancelacion"] = 0

    temp["prioridad_total"] = (
        temp["flag_riesgo_alto"]
        + temp["flag_pricing_critico"]
        + temp["flag_abuso_alto"]
        + temp["flag_cancelacion"]
    )

    sort_cols = ["prioridad_total"]
    ascending = [False]

    if "approved_cost_sum" in temp.columns:
        temp["approved_cost_sum"] = pd.to_numeric(temp["approved_cost_sum"], errors="coerce")
        sort_cols.append("approved_cost_sum")
        ascending.append(False)

    cols = [
        "member_id",
        "policy_id",
        "region",
        "plan_type",
        "plan_tier",
        "predicted_risk_segment",
        "pricing_status",
        "member_abuse_severity",
        "approved_cost_sum",
        "premium_monthly",
        "claims_count",
        "cancellation_flag",
        "prioridad_total",
    ]
    cols = [c for c in cols if c in temp.columns]

    out = temp.sort_values(sort_cols, ascending=ascending)[cols].head(20).copy()
    return out


# =========================================================
# SAFE LOADS
# =========================================================
try:
    df = get_policy_member().copy()
except Exception as e:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.error(f"No se pudo cargar el dataset maestro principal: {e}")
    st.info("Revisa que el archivo de datos esté correctamente incluido en el repositorio y que la ruta de carga sea válida.")
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
    **Visión general ejecutiva del portfolio**  
    Esta página resume de forma integrada la cartera asegurada, la adecuación de pricing,
    la exposición a abuso/fraude y algunos patrones básicos del negocio.

    **Aviso importante:** este dashboard es una herramienta analítica de apoyo a la decisión.
    No sustituye la evaluación actuarial, clínica, antifraude, de suscripción o de negocio.
    Algunos indicadores o primas simuladas pueden diferir de condiciones operativas reales.
    """
)

overview = dataset_overview(df)
n_rows = overview.get("rows", len(df))
n_cols = overview.get("columns", df.shape[1])

# =========================================================
# KPIS GLOBALES
# =========================================================
st.markdown("## Resumen general")

unique_members = safe_count_unique(df, "member_id")
unique_policies = safe_count_unique(df, "policy_id")
provider_n = len(provider_df) if not provider_df.empty else None
prospect_n = len(prospect_df) if not prospect_df.empty else None

avg_premium_monthly = safe_numeric_mean(df, "premium_monthly")
sum_approved_cost = safe_numeric_sum(df, "approved_cost_sum")
avg_loss_ratio = safe_numeric_mean(df, "observed_loss_ratio")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Miembros analizados", fmt_int(unique_members if unique_members is not None else n_rows))
k2.metric("Pólizas analizadas", fmt_int(unique_policies if unique_policies is not None else n_rows))
k3.metric("Prestadores", fmt_int(provider_n))
k4.metric("Prospectos", fmt_int(prospect_n))

k5, k6, k7 = st.columns(3)
k5.metric("Prima mensual media", fmt_float(avg_premium_monthly))
k6.metric("Coste aprobado total", fmt_float(sum_approved_cost))
k7.metric("Siniestralidad observada media", fmt_pct(avg_loss_ratio * 100 if avg_loss_ratio is not None and avg_loss_ratio <= 1.5 else avg_loss_ratio))

# =========================================================
# ALERTAS CLAVE
# =========================================================
st.markdown("## Señales clave de cartera")

total_cases = len(df)

high_risk_n = detect_high_risk_count(df)
underpriced_n = detect_underpriced_count(df)
high_abuse_n = detect_high_abuse_count(df)

a1, a2, a3 = st.columns(3)

a1.metric(
    "Riesgo alto / muy alto",
    fmt_int(high_risk_n),
    fmt_pct(pct(high_risk_n, total_cases)) if high_risk_n is not None else None,
)

a2.metric(
    "Pólizas infraprecificadas",
    fmt_int(underpriced_n),
    fmt_pct(pct(underpriced_n, total_cases)) if underpriced_n is not None else None,
)

a3.metric(
    "Abuso alto / muy alto",
    fmt_int(high_abuse_n),
    fmt_pct(pct(high_abuse_n, total_cases)) if high_abuse_n is not None else None,
)

# =========================================================
# TARJETAS EJECUTIVAS
# =========================================================
st.markdown("## Lectura ejecutiva")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### Riesgo")
    risk_prob_mean = safe_numeric_mean(df, "predicted_risk_probability")
    top_risk_segment = None
    if "predicted_risk_segment" in df.columns:
        mode = df["predicted_risk_segment"].dropna().astype(str).mode()
        if not mode.empty:
            top_risk_segment = mode.iloc[0]

    st.write(f"**Probabilidad media de riesgo:** {fmt_float(risk_prob_mean)}")
    st.write(f"**Segmento dominante:** {top_risk_segment if top_risk_segment else 'N/D'}")
    st.write(f"**Casos altos / muy altos:** {fmt_int(high_risk_n)}")

with c2:
    st.markdown("### Pricing")
    pricing_gap_mean = safe_numeric_mean(df, "premium_gap_vs_expected")
    top_pricing_status = None
    if "pricing_status" in df.columns:
        mode = df["pricing_status"].dropna().astype(str).mode()
        if not mode.empty:
            top_pricing_status = mode.iloc[0]

    suggested_mid_mean = safe_numeric_mean(df, "suggested_premium_mid")

    st.write(f"**Estado dominante:** {top_pricing_status if top_pricing_status else 'N/D'}")
    st.write(f"**Gap medio vs esperado:** {fmt_float(pricing_gap_mean)}")
    st.write(f"**Prima sugerida media:** {fmt_float(suggested_mid_mean)}")

with c3:
    st.markdown("### Abuso / fraude")
    member_abuse_score_mean = safe_numeric_mean(df, "member_abuse_score")
    flagged_provider_cost_sum = safe_numeric_sum(df, "flagged_provider_cost_sum")
    provider_fraud_max_mean = safe_numeric_mean(df, "provider_fraud_score_max")

    st.write(f"**Score medio de abuso miembro:** {fmt_float(member_abuse_score_mean)}")
    st.write(f"**Coste asociado a prestadores marcados:** {fmt_float(flagged_provider_cost_sum)}")
    st.write(f"**Score máximo fraude proveedor (medio):** {fmt_float(provider_fraud_max_mean)}")

# =========================================================
# EDA RESUMIDO
# =========================================================
st.markdown("## Exploración general")

g1, g2 = st.columns(2)

with g1:
    if "predicted_risk_segment" in df.columns:
        st.markdown("#### Distribución por segmento de riesgo")
        risk_dist = safe_value_count_df(df, "predicted_risk_segment", "segmento")
        if not risk_dist.empty:
            st.bar_chart(risk_dist.set_index("segmento"))
        else:
            st.info("No hay datos suficientes para mostrar este gráfico.")
    else:
        st.info("La columna de segmento de riesgo no está disponible.")

with g2:
    if "pricing_status" in df.columns:
        st.markdown("#### Distribución por estado de pricing")
        pricing_dist = safe_value_count_df(df, "pricing_status", "estado")
        if not pricing_dist.empty:
            st.bar_chart(pricing_dist.set_index("estado"))
        else:
            st.info("No hay datos suficientes para mostrar este gráfico.")
    else:
        st.info("La columna de estado de pricing no está disponible.")

g3, g4 = st.columns(2)

with g3:
    group_col = "plan_tier" if "plan_tier" in df.columns else ("plan_type" if "plan_type" in df.columns else None)
    if group_col:
        st.markdown("#### Mix de cartera por plan")
        plan_dist = safe_value_count_df(df, group_col, "plan")
        if not plan_dist.empty:
            st.bar_chart(plan_dist.set_index("plan"))
        else:
            st.info("No hay datos suficientes para mostrar este gráfico.")
    else:
        st.info("No hay columnas de plan disponibles.")

with g4:
    if "region" in df.columns:
        st.markdown("#### Distribución territorial")
        region_dist = safe_value_count_df(df, "region", "region")
        if not region_dist.empty:
            st.bar_chart(region_dist.set_index("region"))
        else:
            st.info("No hay datos suficientes para mostrar este gráfico.")
    else:
        st.info("La columna de región no está disponible.")

# =========================================================
# TABLA RESUMEN DE CASOS PRIORITARIOS
# =========================================================
st.markdown("## Casos prioritarios")
st.caption("Selección orientativa de registros con mayor criticidad combinada entre riesgo, pricing, abuso y cancelación.")

priority_table = build_priority_table(df)

if priority_table.empty:
    st.info("No hay datos suficientes para construir la tabla de casos prioritarios.")
else:
    st.dataframe(priority_table, use_container_width=True, hide_index=True)

# =========================================================
# PIE DE PÁGINA / CONTEXTO DEL DATASET
# =========================================================
st.markdown("## Contexto del dataset")
f1, f2 = st.columns(2)
f1.metric("Filas del maestro", fmt_int(n_rows))
f2.metric("Columnas del maestro", fmt_int(n_cols))
