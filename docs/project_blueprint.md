# NOVAres SecureHealth — Project Blueprint

## Objective
NOVAres SecureHealth is a modular analytics project designed to support:
- risk scoring
- fraud and abuse detection
- pricing adequacy review
- prospect profiling and plan recommendation

## Data layers
- `data/raw/`: source datasets
- `data/processed/`: transformed and analytical datasets
- `data/dashboard/`: final simplified datasets used by Streamlit

## Analytical modules
1. Risk Scoring
2. Fraud / Abuse
3. Pricing
4. Prospect Profiler

## Dashboard philosophy
The dashboard is intentionally lightweight and consumes only dashboard-ready datasets.
It is not meant to reproduce the full notebook pipeline inside Streamlit.
