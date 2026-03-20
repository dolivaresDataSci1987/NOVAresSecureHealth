import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# =========================================================
# IMPORTS DEL PROYECTO
# =========================================================
try:
    from src.data.load_data import load_policy_member_master
except Exception:
    load_policy_member_master = None

try:
    from src.data.load_data import load_provider_master
except Exception:
    load_provider_master = None

try:
    from src.data.load_data import load_claims_corrected
except Exception:
    load_claims_corrected = None


st.set_page_config(page_title="Rentabilidad del seguro", page_icon="💰", layout="wide")


# =========================================================
# HELPERS DE CARGA
# =========================================================
def _read_csv_fallback(filename: str) -> pd.DataFrame:
    """
    Busca el CSV en rutas típicas del proyecto para no romper
    la app si no existe una función de carga específica.
    """
    candidate_paths = [
        ROOT / "data" / filename,
        ROOT / "data" / "processed" / filename,
        ROOT / "data" / "final" / filename,
        ROOT / "datasets" / filename,
        ROOT / filename,
    ]

    for path in candidate_paths:
        if path.exists():
            return pd.read_csv(path)

    raise FileNotFoundError(f"No se encontró el archivo: {filename}")


@st.cache_data
def get_policy_member_data():
    if load_policy_member_master is not None:
        return load_policy_member_master()
    return _read_csv_fallback("dashboard_master_policy_member.csv")


@st.cache_data
def get_provider_data():
    if load_provider_master is not None:
        return load_provider_master()
    return _read_csv_fallback("provider_master.csv")


@st.cache_data
def get_claims_data():
    if load_claims_corrected is not None:
        return load_claims_corrected()
    return _read_csv_fallback("claims_corrected.csv")


# =========================================================
# HELPERS GENERALES
# =========================================================
def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fmt_int(x):
    if pd.isna(x):
        return "N/A"
    return f"{int(round(x)):,}".replace(",", ".")


def fmt_currency(x):
    if pd.isna(x):
        return "N/A"
    return f"${x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.1f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def clean_cat(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .replace({"nan": "No informado", "None": "No informado", "": "No informado"})
        .fillna("No informado")
    )


def metric_card_delta_label(value):
    if pd.isna(value):
        return "N/A"
    if value > 0:
        return "Rentable"
    if value < 0:
        return "Pérdida"
    return "Neutro"


def style_profit_table(df_in: pd.DataFrame):
    df_out = df_in.copy()

    currency_cols = [
        "prima_total",
        "coste_total",
        "beneficio_total",
        "beneficio_medio",
        "prima_atribuida_total",
        "coste_aprobado_total",
        "beneficio_estimado_total",
        "coste_facturado_total",
        "coste_rechazado_total",
    ]
    pct_cols = ["margen_pct", "pct_registros_rentables", "approval_rate_media"]

    for col in currency_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].map(fmt_currency)

    for col in pct_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].map(fmt_pct)

    return df_out


# =========================================================
# CARGA DE DATOS
# =========================================================
try:
    policy_df = get_policy_member_data().copy()
except Exception as e:
    st.error(f"No se pudo cargar dashboard_master_policy_member: {e}")
    st.stop()

try:
    claims_df = get_claims_data().copy()
except Exception:
    claims_df = pd.DataFrame()

try:
    provider_df = get_provider_data().copy()
except Exception:
    provider_df = pd.DataFrame()


# =========================================================
# NORMALIZACIÓN BASE POLICY MEMBER
# =========================================================
required_policy_cols = ["member_id", "policy_id", "plan_type", "premium_annual", "premium_monthly", "approved_cost_sum"]
for col in required_policy_cols:
    if col not in policy_df.columns:
        policy_df[col] = np.nan

policy_df["premium_annual"] = safe_num(policy_df["premium_annual"])
policy_df["premium_monthly"] = safe_num(policy_df["premium_monthly"])
policy_df["approved_cost_sum"] = safe_num(policy_df["approved_cost_sum"])
policy_df["claims_count"] = safe_num(policy_df["claims_count"]) if "claims_count" in policy_df.columns else np.nan
policy_df["claims_n"] = safe_num(policy_df["claims_n"]) if "claims_n" in policy_df.columns else np.nan

# Benefit principal a nivel miembro/póliza
policy_df["prima_anual_calc"] = policy_df["premium_annual"]
policy_df["coste_total_calc"] = policy_df["approved_cost_sum"]
policy_df["beneficio_total_calc"] = policy_df["prima_anual_calc"] - policy_df["coste_total_calc"]

# Número de claims robusto
policy_df["claims_n_calc"] = np.where(
    policy_df["claims_count"].fillna(0) > 0,
    policy_df["claims_count"],
    policy_df["claims_n"]
)
policy_df["claims_n_calc"] = safe_num(pd.Series(policy_df["claims_n_calc"])).fillna(0)

# Prima atribuida por claim para análisis por proveedor / procedimiento
policy_df["prima_atribuida_por_claim"] = np.where(
    policy_df["claims_n_calc"] > 0,
    policy_df["prima_anual_calc"] / policy_df["claims_n_calc"],
    np.nan
)

# Segmento principal para negocio
if "predicted_risk_segment" in policy_df.columns:
    policy_df["segmento_negocio"] = clean_cat(policy_df["predicted_risk_segment"])
elif "region" in policy_df.columns:
    policy_df["segmento_negocio"] = clean_cat(policy_df["region"])
else:
    policy_df["segmento_negocio"] = "No informado"

policy_df["plan_type"] = clean_cat(policy_df["plan_type"]) if "plan_type" in policy_df.columns else "No informado"
policy_df["plan_tier"] = clean_cat(policy_df["plan_tier"]) if "plan_tier" in policy_df.columns else "No informado"
policy_df["coverage_scope"] = clean_cat(policy_df["coverage_scope"]) if "coverage_scope" in policy_df.columns else "No informado"
policy_df["provider_network_type"] = clean_cat(policy_df["provider_network_type"]) if "provider_network_type" in policy_df.columns else "No informado"
policy_df["region"] = clean_cat(policy_df["region"]) if "region" in policy_df.columns else "No informado"


# =========================================================
# ENRIQUECIMIENTO CLAIMS PARA ANÁLISIS POR PROVEEDOR / PROCEDIMIENTO
# =========================================================
claims_enriched = pd.DataFrame()

if not claims_df.empty:
    claims_df = claims_df.copy()

    join_cols = [
        "member_id",
        "policy_id",
        "prima_atribuida_por_claim",
        "prima_anual_calc",
        "plan_type",
        "plan_tier",
        "coverage_scope",
        "provider_network_type",
        "segmento_negocio",
        "region",
    ]
    join_cols = [c for c in join_cols if c in policy_df.columns]

    claims_enriched = claims_df.merge(
        policy_df[join_cols].drop_duplicates(subset=["member_id", "policy_id"]),
        on=["member_id", "policy_id"],
        how="left",
    )

    if "claim_amount_approved" not in claims_enriched.columns:
        claims_enriched["claim_amount_approved"] = np.nan
    if "claim_amount_billed" not in claims_enriched.columns:
        claims_enriched["claim_amount_billed"] = np.nan
    if "claim_amount_rejected" not in claims_enriched.columns:
        claims_enriched["claim_amount_rejected"] = np.nan

    claims_enriched["claim_amount_approved"] = safe_num(claims_enriched["claim_amount_approved"])
    claims_enriched["claim_amount_billed"] = safe_num(claims_enriched["claim_amount_billed"])
    claims_enriched["claim_amount_rejected"] = safe_num(claims_enriched["claim_amount_rejected"])
    claims_enriched["prima_atribuida_por_claim"] = safe_num(claims_enriched["prima_atribuida_por_claim"])

    claims_enriched["beneficio_estimado_claim"] = (
        claims_enriched["prima_atribuida_por_claim"] - claims_enriched["claim_amount_approved"]
    )

    if "service_category" in claims_enriched.columns:
        claims_enriched["service_category"] = clean_cat(claims_enriched["service_category"])
    else:
        claims_enriched["service_category"] = "No informado"

    if "service_type" in claims_enriched.columns:
        claims_enriched["service_type"] = clean_cat(claims_enriched["service_type"])
    else:
        claims_enriched["service_type"] = "No informado"

    if "procedure_code_group" in claims_enriched.columns:
        claims_enriched["procedure_code_group"] = clean_cat(claims_enriched["procedure_code_group"])
    else:
        claims_enriched["procedure_code_group"] = "No informado"

    if "provider_id" in claims_enriched.columns:
        claims_enriched["provider_id"] = clean_cat(claims_enriched["provider_id"])
    else:
        claims_enriched["provider_id"] = "No informado"

    if not provider_df.empty and "provider_id" in provider_df.columns:
        provider_small = provider_df.copy()
        if "provider_name" not in provider_small.columns:
            provider_small["provider_name"] = provider_small["provider_id"]
        if "provider_type" not in provider_small.columns:
            provider_small["provider_type"] = "No informado"
        if "specialty_group" not in provider_small.columns:
            provider_small["specialty_group"] = "No informado"

        provider_small["provider_name"] = clean_cat(provider_small["provider_name"])
        provider_small["provider_type"] = clean_cat(provider_small["provider_type"])
        provider_small["specialty_group"] = clean_cat(provider_small["specialty_group"])

        claims_enriched = claims_enriched.merge(
            provider_small[["provider_id", "provider_name", "provider_type", "specialty_group"]].drop_duplicates(),
            on="provider_id",
            how="left",
        )
    else:
        claims_enriched["provider_name"] = claims_enriched["provider_id"]
        claims_enriched["provider_type"] = "No informado"
        claims_enriched["specialty_group"] = "No informado"

    claims_enriched["provider_name"] = clean_cat(claims_enriched["provider_name"])
    claims_enriched["provider_type"] = clean_cat(claims_enriched["provider_type"])
    claims_enriched["specialty_group"] = clean_cat(claims_enriched["specialty_group"])


# =========================================================
# AGREGADORES
# =========================================================
def summary_member_level(data: pd.DataFrame, group_col: str) -> pd.DataFrame:
    temp = data.copy()
    temp[group_col] = clean_cat(temp[group_col])

    out = (
        temp.groupby(group_col, dropna=False)
        .agg(
            miembros=("member_id", "nunique"),
            polizas=("policy_id", "nunique"),
            prima_total=("prima_anual_calc", "sum"),
            coste_total=("coste_total_calc", "sum"),
            beneficio_total=("beneficio_total_calc", "sum"),
            beneficio_medio=("beneficio_total_calc", "mean"),
        )
        .reset_index()
    )

    out["margen_pct"] = np.where(
        out["prima_total"].fillna(0) != 0,
        (out["beneficio_total"] / out["prima_total"]) * 100,
        np.nan,
    )

    out["pct_registros_rentables"] = (
        temp.groupby(group_col)["beneficio_total_calc"]
        .apply(lambda s: (s > 0).mean() * 100 if len(s) > 0 else np.nan)
        .values
    )

    out = out.sort_values("beneficio_total", ascending=False).reset_index(drop=True)
    return out


def summary_claim_level(data: pd.DataFrame, group_col: str) -> pd.DataFrame:
    temp = data.copy()
    temp[group_col] = clean_cat(temp[group_col])

    out = (
        temp.groupby(group_col, dropna=False)
        .agg(
            claims=("claim_id", "count"),
            miembros=("member_id", "nunique"),
            prima_atribuida_total=("prima_atribuida_por_claim", "sum"),
            coste_aprobado_total=("claim_amount_approved", "sum"),
            coste_facturado_total=("claim_amount_billed", "sum"),
            coste_rechazado_total=("claim_amount_rejected", "sum"),
            beneficio_estimado_total=("beneficio_estimado_claim", "sum"),
            approval_rate_media=("approval_rate", "mean") if "approval_rate" in temp.columns else ("claim_id", "count"),
        )
        .reset_index()
    )

    if "approval_rate_media" in out.columns and out["approval_rate_media"].max() > 1.5:
        out["approval_rate_media"] = out["approval_rate_media"] / 100

    out["margen_pct"] = np.where(
        out["prima_atribuida_total"].fillna(0) != 0,
        (out["beneficio_estimado_total"] / out["prima_atribuida_total"]) * 100,
        np.nan,
    )

    out = out.sort_values("beneficio_estimado_total", ascending=False).reset_index(drop=True)
    return out


def top_bottom(df_in: pd.DataFrame, metric_col: str, n: int = 10):
    if df_in.empty:
        return pd.DataFrame(), pd.DataFrame()
    top_df = df_in.sort_values(metric_col, ascending=False).head(n).copy()
    bottom_df = df_in.sort_values(metric_col, ascending=True).head(n).copy()
    return top_df, bottom_df


# =========================================================
# FILTROS
# =========================================================
st.title("💰 Rentabilidad del seguro")
st.caption(
    "Análisis de dónde gana y pierde dinero la aseguradora por segmento, póliza, cliente, proveedor y tipo de procedimiento."
)

with st.expander("Cómo se calcula la rentabilidad en esta página", expanded=False):
    st.markdown(
        """
**A nivel cliente / póliza**  
- **Prima** = `premium_annual`
- **Coste** = `approved_cost_sum`
- **Beneficio** = `premium_annual - approved_cost_sum`

**A nivel proveedor / procedimiento**  
- Se usa una **estimación atribuida por claim**
- La prima anual del miembro se reparte entre sus claims:  
  **prima atribuida por claim = premium_annual / número de claims**
- Luego:  
  **beneficio estimado del claim = prima atribuida por claim - claim_amount_approved**

Esto es útil para identificar focos de margen y pérdida, aunque no sustituye una contabilidad actuarial completa.
        """
    )

filter_col_1, filter_col_2, filter_col_3, filter_col_4 = st.columns(4)

working_policy = policy_df.copy()

with filter_col_1:
    segment_options = sorted(working_policy["segmento_negocio"].dropna().unique().tolist())
    selected_segments = st.multiselect("Segmento", segment_options, default=segment_options)

with filter_col_2:
    plan_options = sorted(working_policy["plan_type"].dropna().unique().tolist())
    selected_plans = st.multiselect("Tipo de póliza", plan_options, default=plan_options)

with filter_col_3:
    tier_options = sorted(working_policy["plan_tier"].dropna().unique().tolist())
    selected_tiers = st.multiselect("Tier del plan", tier_options, default=tier_options)

with filter_col_4:
    region_options = sorted(working_policy["region"].dropna().unique().tolist())
    selected_regions = st.multiselect("Región", region_options, default=region_options)

working_policy = working_policy[
    working_policy["segmento_negocio"].isin(selected_segments)
    & working_policy["plan_type"].isin(selected_plans)
    & working_policy["plan_tier"].isin(selected_tiers)
    & working_policy["region"].isin(selected_regions)
].copy()

if working_policy.empty:
    st.warning("No hay datos tras aplicar los filtros.")
    st.stop()

working_claims = pd.DataFrame()
if not claims_enriched.empty:
    valid_keys = working_policy[["member_id", "policy_id"]].drop_duplicates()
    working_claims = claims_enriched.merge(valid_keys, on=["member_id", "policy_id"], how="inner")


# =========================================================
# KPIS
# =========================================================
prima_total = working_policy["prima_anual_calc"].sum()
coste_total = working_policy["coste_total_calc"].sum()
beneficio_total = working_policy["beneficio_total_calc"].sum()
margen_total = (beneficio_total / prima_total * 100) if prima_total else np.nan
pct_rentables = (working_policy["beneficio_total_calc"] > 0).mean() * 100 if len(working_policy) else np.nan
beneficio_medio = working_policy["beneficio_total_calc"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Miembros analizados", fmt_int(working_policy["member_id"].nunique()))
k2.metric("Prima total", fmt_currency(prima_total))
k3.metric("Coste total", fmt_currency(coste_total))
k4.metric("Beneficio total", fmt_currency(beneficio_total), delta=metric_card_delta_label(beneficio_total))
k5.metric("Margen sobre prima", fmt_pct(margen_total))

k6, k7, k8 = st.columns(3)
k6.metric("Beneficio medio por miembro", fmt_currency(beneficio_medio))
k7.metric("% de miembros rentables", fmt_pct(pct_rentables))
k8.metric("Claims en filtro", fmt_int(len(working_claims)) if not working_claims.empty else 0)

st.markdown("### Lectura ejecutiva")

if pd.isna(margen_total):
    lectura = "No ha sido posible calcular el margen."
elif margen_total >= 15:
    lectura = "La cartera filtrada muestra una rentabilidad sólida."
elif margen_total >= 5:
    lectura = "La cartera filtrada es rentable, aunque con margen moderado."
elif margen_total >= 0:
    lectura = "La cartera filtrada apenas genera beneficio y necesita ajuste."
else:
    lectura = "La cartera filtrada está en pérdidas y requiere intervención."

st.info(
    f"**Conclusión:** {lectura} "
    f"En el perímetro seleccionado, la aseguradora genera **{fmt_currency(beneficio_total)}** "
    f"de beneficio total, con un margen de **{fmt_pct(margen_total)}** y un **{fmt_pct(pct_rentables)}** "
    f"de miembros rentables."
)


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Visión general",
        "Segmentos y pólizas",
        "Proveedores y procedimientos",
        "Cliente a cliente",
        "Optimización de ganancias",
    ]
)


# =========================================================
# TAB 1
# =========================================================
with tab1:
    st.subheader("Distribución del beneficio por miembro")

    dist = working_policy["beneficio_total_calc"].dropna()
    if not dist.empty:
        bins = pd.cut(dist, bins=20)
        hist = bins.value_counts().sort_index().reset_index()
        hist.columns = ["rango", "miembros"]
        hist["rango"] = hist["rango"].astype(str)
        st.bar_chart(hist.set_index("rango"))
    else:
        st.info("No hay datos suficientes para mostrar la distribución.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Top 20 miembros por beneficio**")
        cols = [
            c for c in [
                "member_id", "policy_id", "segmento_negocio", "plan_type", "plan_tier",
                "prima_anual_calc", "coste_total_calc", "beneficio_total_calc"
            ] if c in working_policy.columns
        ]
        top_members = working_policy[cols].sort_values("beneficio_total_calc", ascending=False).head(20).copy()
        top_members = top_members.rename(
            columns={
                "segmento_negocio": "segmento",
                "plan_type": "tipo_poliza",
                "plan_tier": "tier",
                "prima_anual_calc": "prima_total",
                "coste_total_calc": "coste_total",
                "beneficio_total_calc": "beneficio_total",
            }
        )
        st.dataframe(style_profit_table(top_members), use_container_width=True)

    with c2:
        st.markdown("**Bottom 20 miembros por beneficio**")
        bottom_members = working_policy[cols].sort_values("beneficio_total_calc", ascending=True).head(20).copy()
        bottom_members = bottom_members.rename(
            columns={
                "segmento_negocio": "segmento",
                "plan_type": "tipo_poliza",
                "plan_tier": "tier",
                "prima_anual_calc": "prima_total",
                "coste_total_calc": "coste_total",
                "beneficio_total_calc": "beneficio_total",
            }
        )
        st.dataframe(style_profit_table(bottom_members), use_container_width=True)


# =========================================================
# TAB 2
# =========================================================
with tab2:
    st.subheader("Rentabilidad por segmento")
    seg_summary = summary_member_level(working_policy, "segmento_negocio")
    st.dataframe(style_profit_table(seg_summary), use_container_width=True)

    if not seg_summary.empty:
        st.bar_chart(seg_summary.set_index("segmento_negocio")["beneficio_total"].head(15))

    st.subheader("Rentabilidad por tipo de póliza")
    plan_summary = summary_member_level(working_policy, "plan_type")
    st.dataframe(style_profit_table(plan_summary), use_container_width=True)

    if not plan_summary.empty:
        st.bar_chart(plan_summary.set_index("plan_type")["beneficio_total"].head(15))

    st.subheader("Rentabilidad por tier del plan")
    tier_summary = summary_member_level(working_policy, "plan_tier")
    st.dataframe(style_profit_table(tier_summary), use_container_width=True)

    c1, c2 = st.columns(2)

    top_seg, bottom_seg = top_bottom(seg_summary, "beneficio_total", n=8)
    with c1:
        st.markdown("**Segmentos más rentables**")
        st.dataframe(style_profit_table(top_seg), use_container_width=True)

    with c2:
        st.markdown("**Segmentos menos rentables**")
        st.dataframe(style_profit_table(bottom_seg), use_container_width=True)


# =========================================================
# TAB 3
# =========================================================
with tab3:
    if working_claims.empty:
        st.info(
            "No se pudieron cargar claims para analizar proveedores y procedimientos. "
            "La parte principal de rentabilidad por miembro/póliza sigue disponible."
        )
    else:
        st.subheader("Rentabilidad estimada por proveedor")
        provider_summary = summary_claim_level(working_claims, "provider_name")
        st.dataframe(style_profit_table(provider_summary), use_container_width=True)

        if not provider_summary.empty:
            st.bar_chart(provider_summary.set_index("provider_name")["beneficio_estimado_total"].head(15))

        st.subheader("Rentabilidad estimada por tipo de procedimiento")
        proc_summary = summary_claim_level(working_claims, "service_category")
        st.dataframe(style_profit_table(proc_summary), use_container_width=True)

        if not proc_summary.empty:
            st.bar_chart(proc_summary.set_index("service_category")["beneficio_estimado_total"].head(15))

        st.subheader("Rentabilidad estimada por tipo de servicio")
        service_type_summary = summary_claim_level(working_claims, "service_type")
        st.dataframe(style_profit_table(service_type_summary), use_container_width=True)

        c1, c2 = st.columns(2)

        top_pr, bottom_pr = top_bottom(provider_summary, "beneficio_estimado_total", n=8)
        with c1:
            st.markdown("**Proveedores que más aportan margen estimado**")
            st.dataframe(style_profit_table(top_pr), use_container_width=True)

        with c2:
            st.markdown("**Proveedores con peor resultado estimado**")
            st.dataframe(style_profit_table(bottom_pr), use_container_width=True)


# =========================================================
# TAB 4
# =========================================================
with tab4:
    st.subheader("Análisis cliente a cliente")

    customer_summary = (
        working_policy.groupby("member_id", dropna=False)
        .agg(
            polizas=("policy_id", "nunique"),
            segmento=("segmento_negocio", lambda s: s.mode().iloc[0] if not s.mode().empty else "No informado"),
            tipo_poliza=("plan_type", lambda s: s.mode().iloc[0] if not s.mode().empty else "No informado"),
            tier=("plan_tier", lambda s: s.mode().iloc[0] if not s.mode().empty else "No informado"),
            prima_total=("prima_anual_calc", "sum"),
            coste_total=("coste_total_calc", "sum"),
            beneficio_total=("beneficio_total_calc", "sum"),
            claims=("claims_n_calc", "sum"),
        )
        .reset_index()
    )

    customer_summary["margen_pct"] = np.where(
        customer_summary["prima_total"].fillna(0) != 0,
        (customer_summary["beneficio_total"] / customer_summary["prima_total"]) * 100,
        np.nan,
    )

    sort_options = {
        "Beneficio total (desc)": ("beneficio_total", False),
        "Beneficio total (asc)": ("beneficio_total", True),
        "Margen % (desc)": ("margen_pct", False),
        "Margen % (asc)": ("margen_pct", True),
        "Claims (desc)": ("claims", False),
    }

    sort_label = st.selectbox("Ordenar por", list(sort_options.keys()))
    sort_col, ascending = sort_options[sort_label]

    customer_summary = customer_summary.sort_values(sort_col, ascending=ascending)
    st.dataframe(style_profit_table(customer_summary.head(300)), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Clientes más rentables**")
        st.dataframe(style_profit_table(customer_summary.sort_values("beneficio_total", ascending=False).head(20)), use_container_width=True)

    with c2:
        st.markdown("**Clientes con peor resultado económico**")
        st.dataframe(style_profit_table(customer_summary.sort_values("beneficio_total", ascending=True).head(20)), use_container_width=True)


# =========================================================
# TAB 5
# =========================================================
with tab5:
    st.subheader("Cómo optimizar las ganancias para la aseguradora")

    seg_summary = summary_member_level(working_policy, "segmento_negocio")
    plan_summary = summary_member_level(working_policy, "plan_type")

    best_segment = seg_summary.iloc[0]["segmento_negocio"] if not seg_summary.empty else None
    worst_segment = seg_summary.sort_values("beneficio_total", ascending=True).iloc[0]["segmento_negocio"] if not seg_summary.empty else None

    best_plan = plan_summary.iloc[0]["plan_type"] if not plan_summary.empty else None
    worst_plan = plan_summary.sort_values("beneficio_total", ascending=True).iloc[0]["plan_type"] if not plan_summary.empty else None

    if best_segment:
        st.markdown(
            f"- **Crecimiento selectivo**: reforzar captación y retención en el segmento **{best_segment}**, "
            f"porque concentra una mejor contribución económica."
        )

    if worst_segment:
        st.markdown(
            f"- **Corrección de cartera**: revisar precio, cobertura y utilización del segmento **{worst_segment}**, "
            f"porque está deteriorando la rentabilidad."
        )

    if best_plan:
        st.markdown(
            f"- **Mix de producto**: empujar comercialmente el plan **{best_plan}** si además mantiene estabilidad de siniestralidad."
        )

    if worst_plan:
        st.markdown(
            f"- **Rediseño de póliza**: revisar límites, copagos, franquicias o pricing del plan **{worst_plan}**."
        )

    if not working_claims.empty:
        provider_summary = summary_claim_level(working_claims, "provider_name")
        proc_summary = summary_claim_level(working_claims, "service_category")

        worst_provider = (
            provider_summary.sort_values("beneficio_estimado_total", ascending=True).iloc[0]["provider_name"]
            if not provider_summary.empty else None
        )
        worst_proc = (
            proc_summary.sort_values("beneficio_estimado_total", ascending=True).iloc[0]["service_category"]
            if not proc_summary.empty else None
        )

        if worst_provider:
            st.markdown(
                f"- **Gestión de red**: renegociar tarifas, paquetes o derivación con **{worst_provider}** "
                f"si concentra pérdida estimada recurrente."
            )

        if worst_proc:
            st.markdown(
                f"- **Control de utilización**: revisar autorización previa, indicación clínica y topes en **{worst_proc}**."
            )

    st.markdown(
        "- **Pricing técnico**: trasladar segmentos y pólizas deficitarias a la página de pricing para recalibrar prima técnica."
    )
    st.markdown(
        "- **Retención inteligente**: proteger clientes rentables con buena persistencia y actuar antes sobre cuentas con pérdida sistemática."
    )
    st.markdown(
        "- **Cruce con fraude/abuso**: validar si parte de las pérdidas está explicada por sobreutilización o patrones anómalos."
    )

    p1, p2, p3 = st.columns(3)
    p1.info("**1.** Revisar focos de pérdida estructural por segmento y plan.")
    p2.info("**2.** Actuar sobre proveedores y procedimientos con peor contribución estimada.")
    p3.info("**3.** Ajustar pricing y diseño de cobertura según evidencia económica.")


# =========================================================
# DICCIONARIO
# =========================================================
with st.expander("Diccionario de términos"):
    st.markdown(
        """
**Prima total**: ingreso agregado cobrado por la aseguradora en el perímetro analizado.  

**Coste total**: coste aprobado agregado asociado a la atención sanitaria.  

**Beneficio total**: prima total menos coste total.  

**Margen sobre prima**: beneficio dividido por prima total; sirve para comparar rentabilidad relativa.  

**Miembro rentable**: miembro cuyo beneficio es positivo.  

**Segmento**: agrupación de clientes por riesgo o perfil de negocio.  

**Tipo de póliza**: plan asegurador contratado.  

**Proveedor**: hospital, clínica, centro o profesional que presta el servicio.  

**Tipo de procedimiento / servicio**: categoría asistencial, por ejemplo laboratorio, imagen, consulta, urgencias, etc.  

**Beneficio estimado por claim**: aproximación económica calculada repartiendo la prima del miembro entre sus claims y restando el coste aprobado de cada claim.  

**Optimización de ganancias**: acciones para mejorar margen vía selección de cartera, pricing, cobertura, gestión de red y control de utilización.
        """
    )
