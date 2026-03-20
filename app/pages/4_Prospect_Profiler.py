import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_data import load_policy_member_master

st.set_page_config(page_title="Rentabilidad del seguro", page_icon="💰", layout="wide")


# =========================
# CARGA DE DATOS
# =========================
@st.cache_data
def get_data():
    return load_policy_member_master()


df = get_data().copy()


# =========================
# HELPERS
# =========================
def first_existing_column(dataframe: pd.DataFrame, candidates: list[str]):
    for col in candidates:
        if col in dataframe.columns:
            return col
    return None


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fmt_int(x):
    if pd.isna(x):
        return "N/A"
    return f"{int(round(x, 0)):,}".replace(",", ".")


def fmt_num(x, decimals=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_currency(x):
    if pd.isna(x):
        return "N/A"
    return f"${x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.1f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def normalize_group_values(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .replace({"nan": "No informado", "None": "No informado", "": "No informado"})
        .fillna("No informado")
    )


def build_group_summary(
    dataframe: pd.DataFrame,
    group_col: str,
    profit_col: str,
    premium_col: str | None = None,
    cost_col: str | None = None,
    id_col: str | None = None,
):
    if group_col is None or profit_col is None or group_col not in dataframe.columns:
        return pd.DataFrame()

    temp = dataframe.copy()
    temp[group_col] = normalize_group_values(temp[group_col])
    temp[profit_col] = safe_to_numeric(temp[profit_col])

    agg_dict = {
        profit_col: "sum",
    }

    if premium_col and premium_col in temp.columns:
        temp[premium_col] = safe_to_numeric(temp[premium_col])
        agg_dict[premium_col] = "sum"

    if cost_col and cost_col in temp.columns:
        temp[cost_col] = safe_to_numeric(temp[cost_col])
        agg_dict[cost_col] = "sum"

    summary = temp.groupby(group_col, dropna=False).agg(agg_dict).reset_index()

    counts = temp.groupby(group_col, dropna=False).size().reset_index(name="registros")
    summary = summary.merge(counts, on=group_col, how="left")

    if id_col and id_col in temp.columns:
        unique_ids = (
            temp.groupby(group_col, dropna=False)[id_col]
            .nunique()
            .reset_index(name="clientes_unicos")
        )
        summary = summary.merge(unique_ids, on=group_col, how="left")

    summary = summary.rename(columns={profit_col: "beneficio_total"})

    if premium_col and premium_col in summary.columns:
        summary = summary.rename(columns={premium_col: "prima_total"})

    if cost_col and cost_col in summary.columns:
        summary = summary.rename(columns={cost_col: "coste_total"})

    if "prima_total" in summary.columns:
        summary["margen_pct"] = np.where(
            summary["prima_total"].fillna(0) != 0,
            (summary["beneficio_total"] / summary["prima_total"]) * 100,
            np.nan,
        )
    else:
        summary["margen_pct"] = np.nan

    summary = summary.sort_values("beneficio_total", ascending=False).reset_index(drop=True)
    return summary


def top_bottom_table(summary_df: pd.DataFrame, label_col: str, n: int = 5):
    if summary_df.empty or label_col not in summary_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    top_df = summary_df.head(n).copy()
    bottom_df = summary_df.sort_values("beneficio_total", ascending=True).head(n).copy()
    return top_df, bottom_df


def available_multiselect(container, dataframe: pd.DataFrame, col: str | None, label: str):
    if col is None or col not in dataframe.columns:
        return dataframe

    vals = sorted(
        dataframe[col]
        .dropna()
        .astype(str)
        .replace("", "No informado")
        .unique()
        .tolist()
    )

    if not vals:
        return dataframe

    selected = container.multiselect(label, vals, default=vals)
    if selected:
        dataframe = dataframe[dataframe[col].astype(str).isin(selected)]
    return dataframe


def generate_business_recommendations(
    data: pd.DataFrame,
    segment_col: str | None,
    policy_col: str | None,
    provider_col: str | None,
    procedure_col: str | None,
):
    recs = []

    def pick_best_and_worst(col_name, label):
        if not col_name or col_name not in data.columns or "beneficio_total" not in data.columns:
            return None, None
        valid = data[[col_name, "beneficio_total"]].dropna().copy()
        if valid.empty:
            return None, None
        best = valid.sort_values("beneficio_total", ascending=False).iloc[0]
        worst = valid.sort_values("beneficio_total", ascending=True).iloc[0]
        return best, worst

    if segment_col:
        best, worst = pick_best_and_worst(segment_col, "segmento")
        if best is not None and worst is not None:
            recs.append(
                f"**Segmentación comercial**: reforzar captación y retención en el segmento **{best[segment_col]}**, "
                f"que concentra mayor beneficio, y revisar pricing/cobertura en **{worst[segment_col]}**, "
                f"donde la rentabilidad es más débil."
            )

    if policy_col:
        best, worst = pick_best_and_worst(policy_col, "tipo de póliza")
        if best is not None and worst is not None:
            recs.append(
                f"**Diseño de producto**: potenciar la venta del tipo de póliza **{best[policy_col]}** y revisar "
                f"franquicias, copagos, límites o red de proveedores en **{worst[policy_col]}**."
            )

    if provider_col:
        best, worst = pick_best_and_worst(provider_col, "proveedor")
        if best is not None and worst is not None:
            recs.append(
                f"**Gestión de proveedores**: mantener volumen con el proveedor **{best[provider_col]}** si combina uso y rentabilidad, "
                f"y renegociar tarifas o derivación con **{worst[provider_col]}** si está drenando margen."
            )

    if procedure_col:
        best, worst = pick_best_and_worst(procedure_col, "tipo de procedimiento")
        if best is not None and worst is not None:
            recs.append(
                f"**Control de utilización**: analizar protocolos y autorizaciones previas en **{worst[procedure_col]}**, "
                f"y proteger la eficiencia operativa en **{best[procedure_col]}**."
            )

    recs.append(
        "**Acción transversal**: combinar este análisis con riesgo, fraude/abuso y pricing para diferenciar entre "
        "pérdidas estructurales del producto y pérdidas causadas por utilización intensiva o patrones anómalos."
    )

    return recs


# =========================
# DETECCIÓN DE COLUMNAS
# =========================
member_col = first_existing_column(df, ["member_id", "customer_id", "client_id", "prospect_id"])
policy_col = first_existing_column(df, ["policy_type", "plan_type", "recommended_plan", "product_name", "policy_name"])
segment_col = first_existing_column(df, ["segment", "member_segment", "prospect_segment", "customer_segment"])
provider_col = first_existing_column(df, ["provider_name", "provider_group", "provider_id", "provider"])
procedure_col = first_existing_column(df, ["procedure_category", "service_category", "claim_type", "service_type", "utilization_type"])

premium_col = first_existing_column(
    df,
    [
        "premium_monthly",
        "premium_annual",
        "earned_premium",
        "premium_amount",
        "premium",
        "price",
    ],
)

cost_col = first_existing_column(
    df,
    [
        "claims_cost",
        "total_claim_cost",
        "allowed_amount",
        "medical_cost",
        "cost",
        "base_cost",
        "expected_cost",
    ],
)

profit_col = first_existing_column(
    df,
    [
        "profit",
        "profit_value",
        "net_profit",
        "gross_profit",
        "pricing_profit",
        "margin_value",
        "contribution_margin",
    ],
)

# Si no existe beneficio explícito, lo estimamos con prima - coste
profit_is_estimated = False
working_df = df.copy()

if profit_col is None and premium_col and cost_col:
    working_df["beneficio_estimado"] = safe_to_numeric(working_df[premium_col]) - safe_to_numeric(working_df[cost_col])
    profit_col = "beneficio_estimado"
    profit_is_estimated = True

# Si no existe segmento, intentamos usar archetype si estuviera
if segment_col is None and "archetype" in working_df.columns:
    segment_col = "archetype"

# =========================
# UI - CABECERA
# =========================
st.title("💰 Rentabilidad del seguro")
st.caption(
    "Página orientada a identificar dónde gana y pierde dinero la aseguradora: "
    "segmentos, pólizas, clientes, proveedores y tipos de procedimiento."
)

if profit_is_estimated:
    st.info(
        "El beneficio mostrado en esta página se ha **estimado como prima - coste**, "
        "porque no existe una columna de beneficio explícita en el dataset."
    )

if profit_col is None:
    st.error(
        "No se ha encontrado ninguna columna de beneficio ni una combinación prima/coste que permita estimarlo. "
        "Revisa el dataset cargado en `load_policy_member_master()`."
    )
    st.stop()

# =========================
# FILTROS
# =========================
st.markdown("### Filtros de análisis")

f1, f2, f3, f4 = st.columns(4)

with f1:
    working_df = available_multiselect(f1, working_df, segment_col, "Segmento")

with f2:
    working_df = available_multiselect(f2, working_df, policy_col, "Tipo de póliza")

with f3:
    working_df = available_multiselect(f3, working_df, provider_col, "Proveedor")

with f4:
    working_df = available_multiselect(f4, working_df, procedure_col, "Tipo de procedimiento")

if working_df.empty:
    st.warning("No hay datos tras aplicar los filtros seleccionados.")
    st.stop()

# =========================
# KPIs
# =========================
profit_series = safe_to_numeric(working_df[profit_col])

premium_total = safe_to_numeric(working_df[premium_col]).sum() if premium_col else np.nan
cost_total = safe_to_numeric(working_df[cost_col]).sum() if cost_col else np.nan
profit_total = profit_series.sum()
avg_profit = profit_series.mean()

margin_pct = np.nan
if premium_col:
    if pd.notna(premium_total) and premium_total != 0:
        margin_pct = (profit_total / premium_total) * 100

profitable_share = np.nan
if len(working_df) > 0:
    profitable_share = (profit_series.gt(0).mean()) * 100

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registros analizados", fmt_int(len(working_df)))
k2.metric("Beneficio total", fmt_currency(profit_total))
k3.metric("Prima total", fmt_currency(premium_total) if premium_col else "N/A")
k4.metric("Coste total", fmt_currency(cost_total) if cost_col else "N/A")
k5.metric("Margen sobre prima", fmt_pct(margin_pct) if premium_col else "N/A")

k6, k7 = st.columns(2)
k6.metric("Beneficio medio por registro", fmt_currency(avg_profit))
k7.metric("% de registros rentables", fmt_pct(profitable_share))

# =========================
# RESUMEN EJECUTIVO
# =========================
st.markdown("### Lectura ejecutiva")

mensaje_margen = "No disponible"
if pd.notna(margin_pct):
    if margin_pct >= 15:
        mensaje_margen = "La cartera filtrada muestra una rentabilidad sólida."
    elif margin_pct >= 5:
        mensaje_margen = "La cartera filtrada es rentable, pero con margen moderado."
    elif margin_pct >= 0:
        mensaje_margen = "La cartera apenas genera margen y requiere ajustes."
    else:
        mensaje_margen = "La cartera filtrada está destruyendo valor y necesita intervención."

mensaje_mix = (
    "Esta vista permite detectar dónde conviene crecer, dónde renegociar y dónde restringir uso o red."
)

st.info(
    f"**Conclusión**: {mensaje_margen} "
    f"El beneficio agregado del perímetro analizado es **{fmt_currency(profit_total)}** "
    f"y el porcentaje de registros con resultado positivo es **{fmt_pct(profitable_share)}**. "
    f"{mensaje_mix}"
)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Visión general",
        "Segmentos y pólizas",
        "Proveedores y procedimientos",
        "Cliente a cliente",
        "Optimización de ganancias",
    ]
)

with tab1:
    st.subheader("Distribución del beneficio")
    dist_df = working_df[[profit_col]].copy()
    dist_df[profit_col] = safe_to_numeric(dist_df[profit_col])

    if not dist_df.empty:
        bins = pd.cut(dist_df[profit_col], bins=20)
        hist = bins.value_counts().sort_index().reset_index()
        hist.columns = ["rango_beneficio", "registros"]
        hist["rango_beneficio"] = hist["rango_beneficio"].astype(str)
        st.bar_chart(hist.set_index("rango_beneficio"))

    st.subheader("Top 20 registros por beneficio")
    cols_top = [c for c in [member_col, policy_col, segment_col, provider_col, procedure_col, premium_col, cost_col, profit_col] if c in working_df.columns]
    top_records = working_df[cols_top].copy()
    top_records[profit_col] = safe_to_numeric(top_records[profit_col])
    top_records = top_records.sort_values(profit_col, ascending=False).head(20)
    st.dataframe(top_records, use_container_width=True)

    st.subheader("Bottom 20 registros por beneficio")
    bottom_records = working_df[cols_top].copy()
    bottom_records[profit_col] = safe_to_numeric(bottom_records[profit_col])
    bottom_records = bottom_records.sort_values(profit_col, ascending=True).head(20)
    st.dataframe(bottom_records, use_container_width=True)

with tab2:
    st.subheader("Rentabilidad por segmento")
    if segment_col:
        seg_summary = build_group_summary(
            working_df, segment_col, profit_col, premium_col, cost_col, member_col
        )
        st.dataframe(seg_summary, use_container_width=True)

        seg_chart = seg_summary[[segment_col, "beneficio_total"]].head(15).set_index(segment_col)
        st.bar_chart(seg_chart)

        top_seg, bottom_seg = top_bottom_table(seg_summary, segment_col, n=5)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Segmentos más rentables**")
            st.dataframe(top_seg, use_container_width=True)
        with c2:
            st.markdown("**Segmentos menos rentables**")
            st.dataframe(bottom_seg, use_container_width=True)
    else:
        st.info("No se ha encontrado una columna de segmento en el dataset.")

    st.subheader("Rentabilidad por tipo de póliza")
    if policy_col:
        pol_summary = build_group_summary(
            working_df, policy_col, profit_col, premium_col, cost_col, member_col
        )
        st.dataframe(pol_summary, use_container_width=True)

        pol_chart = pol_summary[[policy_col, "beneficio_total"]].head(15).set_index(policy_col)
        st.bar_chart(pol_chart)

        top_pol, bottom_pol = top_bottom_table(pol_summary, policy_col, n=5)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Pólizas más rentables**")
            st.dataframe(top_pol, use_container_width=True)
        with c4:
            st.markdown("**Pólizas menos rentables**")
            st.dataframe(bottom_pol, use_container_width=True)
    else:
        st.info("No se ha encontrado una columna de tipo de póliza en el dataset.")

with tab3:
    st.subheader("Rentabilidad por proveedor")
    if provider_col:
        provider_summary = build_group_summary(
            working_df, provider_col, profit_col, premium_col, cost_col, member_col
        )
        st.dataframe(provider_summary, use_container_width=True)

        provider_chart = provider_summary[[provider_col, "beneficio_total"]].head(15).set_index(provider_col)
        st.bar_chart(provider_chart)

        top_pr, bottom_pr = top_bottom_table(provider_summary, provider_col, n=5)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Proveedores que más aportan margen**")
            st.dataframe(top_pr, use_container_width=True)
        with c2:
            st.markdown("**Proveedores con peor resultado**")
            st.dataframe(bottom_pr, use_container_width=True)
    else:
        st.info("No se ha encontrado una columna de proveedor en el dataset.")

    st.subheader("Rentabilidad por tipo de procedimiento")
    if procedure_col:
        proc_summary = build_group_summary(
            working_df, procedure_col, profit_col, premium_col, cost_col, member_col
        )
        st.dataframe(proc_summary, use_container_width=True)

        proc_chart = proc_summary[[procedure_col, "beneficio_total"]].head(15).set_index(procedure_col)
        st.bar_chart(proc_chart)

        top_proc, bottom_proc = top_bottom_table(proc_summary, procedure_col, n=5)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Procedimientos más rentables**")
            st.dataframe(top_proc, use_container_width=True)
        with c4:
            st.markdown("**Procedimientos menos rentables**")
            st.dataframe(bottom_proc, use_container_width=True)
    else:
        st.info("No se ha encontrado una columna de tipo de procedimiento en el dataset.")

with tab4:
    st.subheader("Análisis cliente a cliente")

    if member_col:
        customer_summary = build_group_summary(
            working_df, member_col, profit_col, premium_col, cost_col, member_col
        )

        sort_options = {
            "Beneficio total (desc)": ("beneficio_total", False),
            "Beneficio total (asc)": ("beneficio_total", True),
            "Margen % (desc)": ("margen_pct", False),
            "Margen % (asc)": ("margen_pct", True),
            "Registros (desc)": ("registros", False),
        }

        sort_label = st.selectbox("Ordenar tabla por", list(sort_options.keys()))
        sort_col, asc = sort_options[sort_label]
        customer_summary = customer_summary.sort_values(sort_col, ascending=asc)

        st.dataframe(customer_summary.head(300), use_container_width=True)

        st.markdown("**Clientes que más beneficio generan**")
        st.dataframe(customer_summary.sort_values("beneficio_total", ascending=False).head(20), use_container_width=True)

        st.markdown("**Clientes con peor resultado económico**")
        st.dataframe(customer_summary.sort_values("beneficio_total", ascending=True).head(20), use_container_width=True)
    else:
        st.info("No se ha encontrado una columna de identificador de cliente/miembro.")

with tab5:
    st.subheader("Cómo optimizar las ganancias para la aseguradora")

    seg_summary = build_group_summary(working_df, segment_col, profit_col, premium_col, cost_col, member_col) if segment_col else pd.DataFrame()
    pol_summary = build_group_summary(working_df, policy_col, profit_col, premium_col, cost_col, member_col) if policy_col else pd.DataFrame()
    provider_summary = build_group_summary(working_df, provider_col, profit_col, premium_col, cost_col, member_col) if provider_col else pd.DataFrame()
    proc_summary = build_group_summary(working_df, procedure_col, profit_col, premium_col, cost_col, member_col) if procedure_col else pd.DataFrame()

    recommendations = generate_business_recommendations(
        {
            segment_col: seg_summary[segment_col] if segment_col and not seg_summary.empty else pd.Series(dtype=str),
            policy_col: pol_summary[policy_col] if policy_col and not pol_summary.empty else pd.Series(dtype=str),
            provider_col: provider_summary[provider_col] if provider_col and not provider_summary.empty else pd.Series(dtype=str),
            procedure_col: proc_summary[procedure_col] if procedure_col and not proc_summary.empty else pd.Series(dtype=str),
            "beneficio_total": (
                pd.concat(
                    [
                        seg_summary["beneficio_total"] if not seg_summary.empty else pd.Series(dtype=float),
                        pol_summary["beneficio_total"] if not pol_summary.empty else pd.Series(dtype=float),
                        provider_summary["beneficio_total"] if not provider_summary.empty else pd.Series(dtype=float),
                        proc_summary["beneficio_total"] if not proc_summary.empty else pd.Series(dtype=float),
                    ],
                    axis=0,
                    ignore_index=True,
                )
            ),
        },
        segment_col,
        policy_col,
        provider_col,
        procedure_col,
    )

    # Recomendaciones más controladas y claras
    if segment_col and not seg_summary.empty:
        best_seg = seg_summary.sort_values("beneficio_total", ascending=False).iloc[0][segment_col]
        worst_seg = seg_summary.sort_values("beneficio_total", ascending=True).iloc[0][segment_col]
        st.markdown(
            f"- **Segmentos**: crecer en **{best_seg}** y revisar estructura de precio, cobertura y uso en **{worst_seg}**."
        )

    if policy_col and not pol_summary.empty:
        best_pol = pol_summary.sort_values("beneficio_total", ascending=False).iloc[0][policy_col]
        worst_pol = pol_summary.sort_values("beneficio_total", ascending=True).iloc[0][policy_col]
        st.markdown(
            f"- **Producto / póliza**: impulsar la comercialización de **{best_pol}** y rediseñar **{worst_pol}** si presenta margen insuficiente."
        )

    if provider_col and not provider_summary.empty:
        worst_pr = provider_summary.sort_values("beneficio_total", ascending=True).iloc[0][provider_col]
        st.markdown(
            f"- **Proveedores**: renegociar tarifas, paquetes o derivaciones con **{worst_pr}** si concentra pérdidas recurrentes."
        )

    if procedure_col and not proc_summary.empty:
        worst_proc = proc_summary.sort_values("beneficio_total", ascending=True).iloc[0][procedure_col]
        st.markdown(
            f"- **Procedimientos**: revisar autorización previa, indicación clínica o topes de uso en **{worst_proc}**."
        )

    st.markdown(
        "- **Pricing técnico**: usar este panel para detectar colectivos infrapriced y llevar esa señal a la página de pricing."
    )
    st.markdown(
        "- **Gestión de cartera**: proteger clientes rentables con buena persistencia y actuar precozmente sobre clientes con pérdida sistemática."
    )
    st.markdown(
        "- **Uso inteligente de la red**: desplazar volumen hacia combinaciones proveedor-procedimiento con mejor equilibrio entre coste y resultado."
    )
    st.markdown(
        "- **Cruce con fraude/abuso**: validar si parte de la pérdida responde a sobreutilización, abuso de cobertura o patrones anómalos."
    )

    st.markdown("### Prioridades sugeridas")
    prioridad_1 = "Revisar segmentos, pólizas o proveedores con beneficio negativo acumulado."
    prioridad_2 = "Redirigir la captación hacia perfiles y productos que ya muestran rentabilidad demostrada."
    prioridad_3 = "Llevar a comité de negocio una revisión de pricing y red médica basada en evidencia."

    p1, p2, p3 = st.columns(3)
    p1.info(f"**1.** {prioridad_1}")
    p2.info(f"**2.** {prioridad_2}")
    p3.info(f"**3.** {prioridad_3}")

# =========================
# DICCIONARIO
# =========================
with st.expander("Diccionario de términos"):
    st.markdown(
        """
**Beneficio total**: suma del beneficio generado por el conjunto analizado.  
**Prima total**: importe agregado cobrado a los asegurados en el perímetro filtrado.  
**Coste total**: coste sanitario o asistencial asociado al perímetro filtrado.  
**Margen sobre prima**: beneficio dividido por prima total. Permite comparar rentabilidad relativa.  
**Registro rentable**: observación con beneficio positivo.  
**Segmento**: grupo de clientes con características comunes de negocio o perfil.  
**Tipo de póliza**: producto o plan asegurador contratado.  
**Proveedor**: centro, clínica, hospital o profesional que presta el servicio.  
**Tipo de procedimiento**: categoría asistencial como laboratorio, imagen, consulta, urgencias, etc.  
**Optimización de ganancias**: acciones para mejorar margen vía pricing, diseño de cobertura, gestión de red, control de uso y selección de cartera.
        """
    )
