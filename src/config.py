from pathlib import Path

# Root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DASHBOARD_DATA_DIR = DATA_DIR / "dashboard"

# Dashboard files
DASHBOARD_MASTER_POLICY_MEMBER = DASHBOARD_DATA_DIR / "dashboard_master_policy_member.csv"
DASHBOARD_MASTER_PROVIDER = DASHBOARD_DATA_DIR / "dashboard_master_provider.csv"
DASHBOARD_MASTER_PROSPECT = DASHBOARD_DATA_DIR / "dashboard_master_prospect.csv"
DASHBOARD_MASTER_DICTIONARY = DASHBOARD_DATA_DIR / "dashboard_master_dictionary.csv"

# App metadata
APP_TITLE = "NOVAres SecureHealth"
APP_SUBTITLE = "Risk scoring · Fraud/abuse · Pricing · Prospect profiler"
