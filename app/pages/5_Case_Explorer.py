 import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Simulador de Prospecto",
    page_icon="🧭",
    layout="wide"
)


# =========================================================
# CARGA DE DATOS
# =========================================================
@st.cache_data
def load_prospect_data():
    """
    Intenta cargar primero desde la estructura del proyecto.
    Si no existe el loader, hace fallback a CSVs locales/raíz.
    """
    survey_df = None
    pricing_df = None

    # 1) Intentar loader del proyecto
    try:
        from src.data.load_data import load_prospect_master  # type: ignore
        pricing_df = load_prospect_master().copy()
    except Exception:
        pricing_df = None

    # 2) Fallback a CSVs del proyecto o del entorno
    candidate_paths_survey = [
        ROOT / "data" / "processed" / "prospect_survey_synthetic.csv",
        ROOT / "data" / "prospect_survey_synthetic.csv",
        Path("/mnt/data/prospect_survey_synthetic.csv"),
    ]

    candidate_paths_pricing = [
        ROOT / "data" / "processed" / "dashboard_master_prospect.csv",
        ROOT / "data" / "dashboard_master_prospect.csv",
        Path("/mnt/data/dashboard_master_prospect.csv"),
    ]

    if survey_df is None:
        for p in candidate_paths_survey:
            if p.exists():
                survey_df = pd.read_csv(p)
                break

    if pricing_df is None:
        for p in candidate_paths_pricing:
            if p.exists():
                pricing_df = pd.read_csv(p)
                break

    # Si no hay survey, pero sí pricing, devolvemos pricing
    # Si no hay ninguno, error controlado
    if survey_df is None and pricing_df is None:
        raise FileNotFoundError(
            "No se ha podido cargar ni el dataset de prospectos ni el dataset maestro de pricing para prospectos."
        )

    if survey_df is None:
        survey_df = pd.DataFrame()

    if pricing_df is None:
        pricing_df = pd.DataFrame()

    return survey_df, pricing_df


survey_df, pricing_df = load_prospect_data()


# =========================================================
# HELPERS
# =========================================================
def safe_mode(series: pd.Series, default="N/D"):
    s = series.dropna()
    if s.empty:
        return default
    mode_vals = s.mode()
    return mode_vals.iloc[0] if not mode_vals.empty else default


def fmt_money(x):
    try:
        return f"USD {float(x):,.2f}"
    except Exception:
        return "N/D"


def age_to_band(age: int) -> str:
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    return "65+"


def build_base_dataset(survey: pd.DataFrame, pricing: pd.DataFrame) -> pd.DataFrame:
    """
    Une survey y pricing si ambos existen.
    """
    if not survey.empty and not pricing.empty and "prospect_id" in survey.columns and "prospect_id" in pricing.columns:
        merged = survey.merge(pricing, on="prospect_id", how="left", suffixes=("", "_pricing"))
        return merged

    if not survey.empty:
        return survey.copy()

    return pricing.copy()


base_df = build_base_dataset(survey_df, pricing_df)


def get_options(df: pd.DataFrame, col: str, fallback: list[str]) -> list[str]:
    if col in df.columns:
        vals = sorted(df[col].dropna().astype(str).unique().tolist())
        return vals if vals else fallback
    return fallback


age_band_options = get_options(base_df, "age_band", ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"])
sex_options = get_options(base_df, "sex", ["F", "M"])
region_options = get_options(
    base_df,
    "region",
    ["Asunción", "Central", "Alto Paraná", "Itapúa", "Caaguazú", "Cordillera", "Paraguarí", "San Pedro"]
)
health_options = get_options(base_df, "self_rated_health", ["Excellent", "Good", "Fair", "Poor"])
visits_options = get_options(base_df, "visits_12m_band", ["0-1", "2-4", "5-9", "10+"])
er_options = get_options(base_df, "er_visits_12m_band", ["0", "1", "2+"])
bmi_options = get_options(base_df, "bmi_group", ["Normal", "Overweight", "Obese"])
activity_options = get_options(base_df, "physical_activity_level", ["High", "Medium", "Low"])
preventive_options = get_options(base_df, "preventive_mindset", ["High", "Medium", "Low"])
preference_options = get_options(base_df, "price_vs_coverage_preference", ["Price", "Balanced", "Coverage"])
copay_options = get_options(base_df, "copay_tolerance", ["High", "Medium", "Low"])
network_options = get_options(base_df, "network_preference", ["Narrow", "Balanced", "Broad"])


RISK_SCORE_MAP = {
    "Low": 1,
    "Moderate": 2,
    "High": 3,
    "Very High": 4,
}

UTIL_SCORE_MAP = {
    "Very Low": 1,
    "Low": 2,
    "Medium": 3,
    "High": 4,
}

HEALTH_SCORE = {
    "Excellent": 0,
    "Good": 1,
    "Fair": 2,
    "Poor": 3,
}

VISITS_SCORE = {
    "0-1": 0,
    "2-4": 1,
    "5-9": 2,
    "10+": 3,
}

ER_SCORE = {
    "0": 0,
    "1": 1,
    "2+": 2,
}

BMI_SCORE = {
    "Normal": 0,
    "Overweight": 1,
    "Obese": 2,
}

ACTIVITY_SCORE = {
    "High": 0,
    "Medium": 1,
    "Low": 2,
}


def score_risk_from_inputs(inputs: dict) -> int:
    score = 0

    age = int(inputs["age"])
    if age >= 65:
        score += 4
    elif age >= 55:
        score += 3
    elif age >= 45:
        score += 2
    elif age >= 35:
        score += 1

    score += int(inputs["dependents_n"]) * 0.3
    score += HEALTH_SCORE.get(inputs["self_rated_health"], 1)
    score += int(inputs["chronic_condition_flag"]) * 2
    score += int(inputs["chronic_condition_count"]) * 1.2
    score += int(inputs["recurrent_medication_flag"]) * 1.5
    score += VISITS_SCORE.get(inputs["visits_12m_band"], 1)
    score += ER_SCORE.get(inputs["er_visits_12m_band"], 0) * 1.5
    score += int(inputs["hospitalization_24m_flag"]) * 3
    score += int(inputs["smoker_flag"]) * 2
    score += BMI_SCORE.get(inputs["bmi_group"], 0)
    score += ACTIVITY_SCORE.get(inputs["physical_activity_level"], 1)

    if int(inputs["maternity_interest_flag"]) == 1:
        score += 1

    if int(inputs["pharmacy_need_flag"]) == 1:
        score += 1

    if int(inputs["chronic_program_interest_flag"]) == 1:
        score += 1

    return round(score)


def numeric_risk_to_tier(score: int) -> str:
    if score <= 4:
        return "Low"
    if score <= 8:
        return "Moderate"
    if score <= 12:
        return "High"
    return "Very High"


def numeric_risk_to_utilization(score: int) -> str:
    if score <= 3:
        return "Very Low"
    if score <= 6:
        return "Low"
    if score <= 10:
        return "Medium"
    return "High"


def estimate_cost_annual(risk_tier: str, utilization_band: str) -> float:
    risk_base = {
        "Low": 650,
        "Moderate": 1150,
        "High": 2200,
        "Very High": 3800,
    }
    util_mult = {
        "Very Low": 0.80,
        "Low": 1.00,
        "Medium": 1.25,
        "High": 1.60,
    }
    return float(risk_base.get(risk_tier, 1150) * util_mult.get(utilization_band, 1.0))


def recommend_plan(inputs: dict, risk_tier: str, utilization_band: str) -> tuple[str, str, str]:
    """
    Devuelve:
    - tipo de plan
    - alcance de cobertura
    - nivel comercial
    """
    preference = inputs["price_vs_coverage_preference"]
    chronic = int(inputs["chronic_condition_flag"])
    maternity = int(inputs["maternity_interest_flag"])
    network = inputs["network_preference"]

    if chronic == 1 and risk_tier in ["High", "Very High"]:
        return "Chronic Care", "Integral", "Premium"

    if maternity == 1 and int(inputs["dependents_n"]) >= 1:
        return "Family Plus", "Integral", "Premium"

    if risk_tier == "Very High":
        return "High Protection", "Hospitalaria + Ambulatoria", "Premium"

    if utilization_band == "High":
        return "Managed Review", "Ambulatoria + control de uso", "Mid"

    if preference == "Price" and risk_tier in ["Low", "Moderate"]:
        return "Essential", "Ambulatoria", "Basic"

    if preference == "Coverage" or network == "Broad":
        return "Standard", "Hospitalaria + Ambulatoria", "Mid"

    return "Standard", "Ambulatoria ampliada", "Mid"


def estimate_premium_monthly(cost_annual: float, plan_level: str) -> float:
    margin_factor = {
        "Basic": 1.18,
        "Mid": 1.28,
        "Premium": 1.40,
    }
    annual_premium = cost_annual * margin_factor.get(plan_level, 1.28)
    return annual_premium / 12


def build_similarity_flags(df: pd.DataFrame, user_row: dict) -> pd.Series:
    """
    Score simple de similitud. Cuantos más puntos, más parecido.
    """
    sim = pd.Series(0, index=df.index, dtype="float64")

    if "age_band" in df.columns:
        sim += (df["age_band"].astype(str) == str(user_row["age_band"])).astype(int) * 2
    if "sex" in df.columns:
        sim += (df["sex"].astype(str) == str(user_row["sex"])).astype(int) * 1
    if "region" in df.columns:
        sim += (df["region"].astype(str) == str(user_row["region"])).astype(int) * 1
    if "dependents_n" in df.columns:
        sim += (df["dependents_n"].fillna(-999).astype(int) == int(user_row["dependents_n"])).astype(int) * 1
    if "self_rated_health" in df.columns:
        sim += (df["self_rated_health"].astype(str) == str(user_row["self_rated_health"])).astype(int) * 2
    if "chronic_condition_flag" in df.columns:
        sim += (df["chronic_condition_flag"].fillna(0).astype(int) == int(user_row["chronic_condition_flag"])).astype(int) * 2
    if "recurrent_medication_flag" in df.columns:
        sim += (df["recurrent_medication_flag"].fillna(0).astype(int) == int(user_row["recurrent_medication_flag"])).astype(int) * 1
    if "visits_12m_band" in df.columns:
        sim += (df["visits_12m_band"].astype(str) == str(user_row["visits_12m_band"])).astype(int) * 2
    if "er_visits_12m_band" in df.columns:
        sim += (df["er_visits_12m_band"].astype(str) == str(user_row["er_visits_12m_band"])).astype(int) * 2
    if "hospitalization_24m_flag" in df.columns:
        sim += (df["hospitalization_24m_flag"].fillna(0).astype(int) == int(user_row["hospitalization_24m_flag"])).astype(int) * 2
    if "smoker_flag" in df.columns:
        sim += (df["smoker_flag"].fillna(0).astype(int) == int(user_row["smoker_flag"])).astype(int) * 2
    if "bmi_group" in df.columns:
        sim += (df["bmi_group"].astype(str) == str(user_row["bmi_group"])).astype(int) * 1
    if "physical_activity_level" in df.columns:
        sim += (df["physical_activity_level"].astype(str) == str(user_row["physical_activity_level"])).astype(int) * 1
    if "network_preference" in df.columns:
        sim += (df["network_preference"].astype(str) == str(user_row["network_preference"])).astype(int) * 1

    return sim


def cohort_from_inputs(df: pd.DataFrame, user_row: dict, top_n: int = 250) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    tmp = df.copy()
    tmp["_sim"] = build_similarity_flags(tmp, user_row)
    tmp = tmp.sort_values("_sim", ascending=False)

    if tmp["_sim"].max() <= 0:
        return tmp.head(min(top_n, len(tmp))).copy()

    return tmp.head(min(top_n, len(tmp))).copy()


def get_driver_texts(inputs: dict, risk_tier: str, utilization_band: str) -> list[str]:
    drivers = []

    if int(inputs["chronic_condition_flag"]) == 1:
        drivers.append("Presencia de condición crónica declarada")
    if int(inputs["chronic_condition_count"]) >= 2:
        drivers.append("Múltiples condiciones crónicas")
    if int(inputs["recurrent_medication_flag"]) == 1:
        drivers.append("Necesidad recurrente de medicación")
    if inputs["self_rated_health"] in ["Fair", "Poor"]:
        drivers.append("Autopercepción de salud desfavorable")
    if inputs["visits_12m_band"] in ["5-9", "10+"]:
        drivers.append("Historial de uso médico relativamente alto")
    if inputs["er_visits_12m_band"] in ["1", "2+"]:
        drivers.append("Uso reciente de urgencias")
    if int(inputs["hospitalization_24m_flag"]) == 1:
        drivers.append("Antecedente de hospitalización reciente")
    if int(inputs["smoker_flag"]) == 1:
        drivers.append("Tabaquismo declarado")
    if inputs["bmi_group"] == "Obese":
        drivers.append("Perfil antropométrico con mayor exposición clínica")
    if inputs["physical_activity_level"] == "Low":
        drivers.append("Baja actividad física")
    if int(inputs["maternity_interest_flag"]) == 1:
        drivers.append("Interés en cobertura de maternidad")
    if int(inputs["pharmacy_need_flag"]) == 1:
        drivers.append("Necesidad de cobertura farmacéutica")
    if int(inputs["chronic_program_interest_flag"]) == 1:
        drivers.append("Conveniencia de programa de gestión crónica")

    if not drivers:
        drivers.append("Perfil global compatible con uso médico contenido")

    if risk_tier in ["High", "Very High"] and utilization_band == "High":
        drivers.append("Se recomienda una propuesta con control técnico y precio consistente con el riesgo")

    return drivers[:5]


# =========================================================
# UI
# =========================================================
st.title("Simulador de Prospecto")

st.markdown(
    """
    Esta página permite al **agente de la aseguradora** registrar un pequeño cuestionario
    previo a la contratación y obtener una **estimación orientativa** del perfil del prospecto:
    **riesgo esperado**, **uso sanitario probable**, **coste potencial para la aseguradora**
    y **tipo de póliza recomendado**.
    """
)

with st.expander("Aviso importante", expanded=False):
    st.info(
        """
        Esta simulación es una **herramienta de apoyo comercial y técnico**.
        No sustituye la suscripción definitiva, la validación actuarial, la revisión médica
        ni las reglas internas de underwriting de la aseguradora.
        """
    )

# =========================================================
# LAYOUT PRINCIPAL
# =========================================================
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Cuestionario del prospecto")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Edad", min_value=18, max_value=100, value=35, step=1)
    with c2:
        sex = st.selectbox("Sexo", sex_options)
    with c3:
        region = st.selectbox("Región", region_options)

    c4, c5, c6 = st.columns(3)
    with c4:
        dependents_n = st.number_input("Número de dependientes", min_value=0, max_value=10, value=0, step=1)
    with c5:
        self_rated_health = st.selectbox("Estado de salud autopercibido", health_options)
    with c6:
        chronic_condition_flag = st.selectbox(
            "¿Tiene condición crónica?",
            options=[0, 1],
            format_func=lambda x: "Sí" if x == 1 else "No"
        )

    c7, c8, c9 = st.columns(3)
    with c7:
        chronic_condition_count = st.number_input("Nº de condiciones crónicas", min_value=0, max_value=10, value=0, step=1)
    with c8:
        recurrent_medication_flag = st.selectbox(
            "¿Usa medicación recurrente?",
            options=[0, 1],
            format_func=lambda x: "Sí" if x == 1 else "No"
        )
    with c9:
        visits_12m_band = st.selectbox("Consultas médicas últimos 12 meses", visits_options)

    c10, c11, c12 = st.columns(3)
    with c10:
        er_visits_12m_band = st.selectbox("Urgencias últimos 12 meses", er_options)
    with c11:
        hospitalization_24m_flag = st.selectbox(
            "¿Hospitalización en 24 meses?",
            options=[0, 1],
            format_func=lambda x: "Sí" if x == 1 else "No"
        )
    with c12:
        smoker_flag = st.selectbox(
            "¿Fumador/a?",
            options=[0, 1],
            format_func=lambda x: "Sí" if x == 1 else "No"
        )

    c13, c14, c15 = st.columns(3)
    with c13:
        bmi_group = st.selectbox("Grupo de IMC", bmi_options)
    with c14:
        physical_activity_level = st.selectbox("Actividad física", activity_options)
    with c15:
        preventive_mindset = st.selectbox("Orientación preventiva", preventive_options)

    c16, c17, c18 = st.columns(3)
    with c16:
        price_vs_coverage_preference = st.selectbox("Preferencia precio vs cobertura", preference_options)
    with c17:
        copay_tolerance = st.selectbox("Tolerancia al copago", copay_options)
    with c18:
        network_preference = st.selectbox("Preferencia de red médica", network_options)

    c19, c20, c21 = st.columns(3)
    with c19:
        maternity_interest_flag = st.selectbox(
            "Interés en maternidad",
            options=[0, 1],
            format_func=lambda x: "Sí" if x == 1 else "No"
        )
    with c20:
        pharmacy_need_flag = st.selectbox(
            "Necesidad de farmacia",
            options=[0, 1],
            format_func=lambda x: "Sí" if x == 1 else "No"
        )
    with c21:
        chronic_program_interest_flag = st.selectbox(
            "Interés en programa crónico",
            options=[0, 1],
            format_func=lambda x: "Sí" if x == 1 else "No"
        )

with right:
    st.subheader("Resumen de captura")
    age_band = age_to_band(int(age))

    summary_df = pd.DataFrame(
        {
            "Campo": [
                "Edad",
                "Banda de edad",
                "Sexo",
                "Región",
                "Dependientes",
                "Salud autopercibida",
                "Crónico",
                "Visitas 12m",
                "Urgencias 12m",
                "Fumador/a",
                "IMC",
                "Actividad física",
            ],
            "Valor": [
                age,
                age_band,
                sex,
                region,
                dependents_n,
                self_rated_health,
                "Sí" if chronic_condition_flag == 1 else "No",
                visits_12m_band,
                er_visits_12m_band,
                "Sí" if smoker_flag == 1 else "No",
                bmi_group,
                physical_activity_level,
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.caption(
        "La estimación final se apoya en reglas simples y en comparación con cohortes sintéticas similares del dataset de prospectos."
    )


# =========================================================
# CÁLCULO
# =========================================================
inputs = {
    "age": age,
    "age_band": age_band,
    "sex": sex,
    "region": region,
    "dependents_n": dependents_n,
    "self_rated_health": self_rated_health,
    "chronic_condition_flag": chronic_condition_flag,
    "chronic_condition_count": chronic_condition_count,
    "recurrent_medication_flag": recurrent_medication_flag,
    "visits_12m_band": visits_12m_band,
    "er_visits_12m_band": er_visits_12m_band,
    "hospitalization_24m_flag": hospitalization_24m_flag,
    "smoker_flag": smoker_flag,
    "bmi_group": bmi_group,
    "physical_activity_level": physical_activity_level,
    "preventive_mindset": preventive_mindset,
    "price_vs_coverage_preference": price_vs_coverage_preference,
    "copay_tolerance": copay_tolerance,
    "network_preference": network_preference,
    "maternity_interest_flag": maternity_interest_flag,
    "pharmacy_need_flag": pharmacy_need_flag,
    "chronic_program_interest_flag": chronic_program_interest_flag,
}

risk_score = score_risk_from_inputs(inputs)
predicted_risk_tier = numeric_risk_to_tier(risk_score)
predicted_utilization_band = numeric_risk_to_utilization(risk_score)
estimated_cost_annual = estimate_cost_annual(predicted_risk_tier, predicted_utilization_band)

recommended_plan, recommended_scope, recommended_level = recommend_plan(
    inputs, predicted_risk_tier, predicted_utilization_band
)

recommended_premium_monthly = estimate_premium_monthly(estimated_cost_annual, recommended_level)
recommended_premium_annual = recommended_premium_monthly * 12

cohort_df = cohort_from_inputs(base_df, inputs, top_n=250)

# Ajustes con cohorte real si existe info suficiente
if not cohort_df.empty:
    if "risk_tier" in cohort_df.columns:
        cohort_mode_risk = safe_mode(cohort_df["risk_tier"], predicted_risk_tier)
        if cohort_mode_risk in ["Low", "Moderate", "High", "Very High"]:
            predicted_risk_tier = cohort_mode_risk

    if "expected_utilization_band" in cohort_df.columns:
        cohort_mode_util = safe_mode(cohort_df["expected_utilization_band"], predicted_utilization_band)
        if cohort_mode_util in ["Very Low", "Low", "Medium", "High"]:
            predicted_utilization_band = cohort_mode_util

    if "recommended_plan" in cohort_df.columns:
        recommended_plan = safe_mode(cohort_df["recommended_plan"], recommended_plan)

    if "recommended_plan_tier" in cohort_df.columns:
        recommended_level = safe_mode(cohort_df["recommended_plan_tier"], recommended_level)

    if "recommended_coverage_scope" in cohort_df.columns:
        recommended_scope = safe_mode(cohort_df["recommended_coverage_scope"], recommended_scope)

    if "recommended_premium_monthly" in cohort_df.columns:
        cohort_premium = pd.to_numeric(cohort_df["recommended_premium_monthly"], errors="coerce").dropna()
        if not cohort_premium.empty:
            recommended_premium_monthly = float(cohort_premium.median())
            recommended_premium_annual = recommended_premium_monthly * 12

# Recalcular coste técnico estimado como base del precio recomendado
margin_map = {"Basic": 1.18, "Mid": 1.28, "Premium": 1.40}
estimated_cost_annual = recommended_premium_annual / margin_map.get(recommended_level, 1.28)

# Métrica sencilla de margen técnico esperado
expected_margin_annual = recommended_premium_annual - estimated_cost_annual
expected_loss_ratio = (estimated_cost_annual / recommended_premium_annual) if recommended_premium_annual > 0 else None

drivers = get_driver_texts(inputs, predicted_risk_tier, predicted_utilization_band)


# =========================================================
# RESULTADOS
# =========================================================
st.markdown("---")
st.subheader("Resultado de la simulación")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Riesgo esperado", predicted_risk_tier)
k2.metric("Uso sanitario esperado", predicted_utilization_band)
k3.metric("Coste anual estimado", fmt_money(estimated_cost_annual))
k4.metric("Prima mensual sugerida", fmt_money(recommended_premium_monthly))
k5.metric("Plan recomendado", recommended_plan)

a1, a2, a3 = st.columns(3)
with a1:
    st.metric("Cobertura sugerida", recommended_scope)
with a2:
    st.metric("Nivel comercial", recommended_level)
with a3:
    st.metric(
        "Margen técnico anual estimado",
        fmt_money(expected_margin_annual)
    )

# =========================================================
# LECTURA DE NEGOCIO
# =========================================================
left2, right2 = st.columns([1.1, 1])

with left2:
    st.subheader("Lectura de negocio")

    if predicted_risk_tier in ["Low", "Moderate"] and predicted_utilization_band in ["Very Low", "Low"]:
        st.success(
            f"""
            **Perfil comercialmente atractivo.**  
            El prospecto muestra una probabilidad relativamente contenida de generar alto coste sanitario.
            Puede ser buen candidato para una propuesta **{recommended_plan}** con foco en captación rentable,
            especialmente si la aseguradora desea crecer en perfiles de menor siniestralidad.
            """
        )
    elif predicted_risk_tier in ["High", "Very High"]:
        st.warning(
            f"""
            **Perfil con presión técnica relevante.**  
            El prospecto presenta señales compatibles con mayor utilización y mayor coste esperado.
            Conviene ofrecer una solución **{recommended_plan}** con precio disciplinado,
            condiciones claras y revisión técnica si la política comercial de la aseguradora lo requiere.
            """
        )
    else:
        st.info(
            f"""
            **Perfil intermedio.**  
            El prospecto podría encajar en una oferta **{recommended_plan}** equilibrada,
            combinando competitividad comercial con protección razonable frente a costes futuros.
            """
        )

    st.markdown("**Principales factores que empujan la recomendación:**")
    for d in drivers:
        st.markdown(f"- {d}")

with right2:
    st.subheader("Indicadores técnicos")
    tech_df = pd.DataFrame(
        {
            "Indicador": [
                "Prima anual sugerida",
                "Coste anual esperado",
                "Margen técnico anual",
                "Loss ratio esperado",
                "Tamaño cohorte comparativa",
            ],
            "Valor": [
                fmt_money(recommended_premium_annual),
                fmt_money(estimated_cost_annual),
                fmt_money(expected_margin_annual),
                f"{expected_loss_ratio:.1%}" if expected_loss_ratio is not None else "N/D",
                f"{len(cohort_df):,}",
            ],
        }
    )
    st.dataframe(tech_df, use_container_width=True, hide_index=True)

    st.caption(
        "El loss ratio esperado se interpreta como coste esperado / prima sugerida. "
        "Cuanto más alto, más ajustado queda el negocio para la aseguradora."
    )


# =========================================================
# TABLAS DE APOYO
# =========================================================
tab1, tab2, tab3 = st.tabs(
    ["Ficha del prospecto", "Cohorte comparable", "Diccionario de términos"]
)

with tab1:
    st.subheader("Ficha completa del prospecto simulado")

    result_row = {
        **inputs,
        "predicted_risk_score": risk_score,
        "predicted_risk_tier": predicted_risk_tier,
        "predicted_utilization_band": predicted_utilization_band,
        "recommended_plan": recommended_plan,
        "recommended_coverage_scope": recommended_scope,
        "recommended_plan_tier": recommended_level,
        "estimated_cost_annual": round(float(estimated_cost_annual), 2),
        "recommended_premium_monthly": round(float(recommended_premium_monthly), 2),
        "recommended_premium_annual": round(float(recommended_premium_annual), 2),
        "expected_margin_annual": round(float(expected_margin_annual), 2),
        "expected_loss_ratio": round(float(expected_loss_ratio), 4) if expected_loss_ratio is not None else None,
    }

    result_df = pd.DataFrame(list(result_row.items()), columns=["campo", "valor"])
    st.dataframe(result_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Muestra de prospectos comparables")

    if cohort_df.empty:
        st.info("No hay cohorte comparable disponible con los datasets cargados.")
    else:
        cols_show = [
            c for c in [
                "prospect_id",
                "age",
                "age_band",
                "sex",
                "region",
                "dependents_n",
                "self_rated_health",
                "chronic_condition_flag",
                "visits_12m_band",
                "er_visits_12m_band",
                "smoker_flag",
                "bmi_group",
                "physical_activity_level",
                "risk_tier",
                "expected_utilization_band",
                "recommended_plan",
                "recommended_plan_tier",
                "recommended_coverage_scope",
                "recommended_premium_monthly",
            ] if c in cohort_df.columns
        ]

        display_df = cohort_df[cols_show].copy()

        if "recommended_premium_monthly" in display_df.columns:
            display_df["recommended_premium_monthly"] = pd.to_numeric(
                display_df["recommended_premium_monthly"], errors="coerce"
            ).round(2)

        st.dataframe(display_df.head(50), use_container_width=True, hide_index=True)

        c1, c2, c3, c4 = st.columns(4)

        if "risk_tier" in cohort_df.columns:
            c1.metric("Riesgo más frecuente", safe_mode(cohort_df["risk_tier"]))
        if "expected_utilization_band" in cohort_df.columns:
            c2.metric("Uso más frecuente", safe_mode(cohort_df["expected_utilization_band"]))
        if "recommended_plan" in cohort_df.columns:
            c3.metric("Plan más frecuente", safe_mode(cohort_df["recommended_plan"]))
        if "recommended_premium_monthly" in cohort_df.columns:
            premium_med = pd.to_numeric(cohort_df["recommended_premium_monthly"], errors="coerce").median()
            c4.metric("Prima mediana cohorte", fmt_money(premium_med))

with tab3:
    st.subheader("Diccionario de términos")

    dict_rows = [
        {
            "término": "Riesgo esperado",
            "definición": "Estimación orientativa de la probabilidad de que el prospecto genere eventos de salud o costes relevantes."
        },
        {
            "término": "Uso sanitario esperado",
            "definición": "Nivel probable de utilización de consultas, urgencias, farmacia u hospitalización."
        },
        {
            "término": "Coste anual estimado",
            "definición": "Estimación aproximada del coste que este prospecto podría suponer para la aseguradora en un año."
        },
        {
            "término": "Prima mensual sugerida",
            "definición": "Precio orientativo recomendado para sostener el riesgo esperado y el nivel de cobertura propuesto."
        },
        {
            "término": "Margen técnico anual estimado",
            "definición": "Diferencia aproximada entre la prima anual sugerida y el coste anual esperado."
        },
        {
            "término": "Loss ratio esperado",
            "definición": "Relación entre coste esperado y prima. Un valor más alto implica un negocio más ajustado para la aseguradora."
        },
        {
            "término": "Plan recomendado",
            "definición": "Tipología de póliza sugerida según el perfil clínico, de uso y de preferencia comercial del prospecto."
        },
        {
            "término": "Cobertura sugerida",
            "definición": "Alcance recomendado de la cobertura: ambulatoria, hospitalaria o integral."
        },
        {
            "término": "Nivel comercial",
            "definición": "Nivel simplificado de oferta: Basic, Mid o Premium."
        },
        {
            "término": "Cohorte comparable",
            "definición": "Grupo de prospectos del dataset con características similares usado como referencia para orientar la recomendación."
        },
    ]

    dict_df = pd.DataFrame(dict_rows)
    st.dataframe(dict_df, use_container_width=True, hide_index=True)
