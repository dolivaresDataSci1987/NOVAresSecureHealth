# NOVAres SecureHealth

NOVAres SecureHealth is a modular analytics project focused on:
- risk scoring
- fraud and abuse detection
- pricing adequacy
- prospect profiling and plan recommendation

## Repository structure

- `data/`: raw, processed, and dashboard-ready datasets
- `notebooks/`: analytical pipeline notebooks
- `src/`: reusable configuration and data utilities
- `app/`: Streamlit dashboard
- `docs/`: project documentation

## Dashboard datasets
The Streamlit app reads only simplified dashboard-ready datasets from:

`data/dashboard/`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/Home.py
