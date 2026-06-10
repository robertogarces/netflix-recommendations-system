from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent

CONFIG_PATH         = ROOT_PATH / "config"
RAW_DATA_PATH       = ROOT_PATH / "data" / "raw"
PROCESSED_DATA_PATH = ROOT_PATH / "data" / "processed"
FINAL_DATA_PATH     = ROOT_PATH / "data" / "final"
MODELS_PATH         = ROOT_PATH / "models"
ARTIFACTS_PATH      = ROOT_PATH / "artifacts"
