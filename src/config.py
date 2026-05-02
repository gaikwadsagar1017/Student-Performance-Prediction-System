from pathlib import Path

# Base project path
BASE_DIR = Path(__file__).resolve().parent.parent

# Folders
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

MODELS_DIR = BASE_DIR / "models"

OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
REPORT_DIR = OUTPUT_DIR / "reports"
PREDICTION_DIR = OUTPUT_DIR / "predictions"


def create_folders():
    folders = [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        SYNTHETIC_DIR,
        MODELS_DIR,
        OUTPUT_DIR,
        CHART_DIR,
        REPORT_DIR,
        PREDICTION_DIR,
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)