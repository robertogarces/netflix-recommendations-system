from pathlib import Path

# Get the route of the actual directory
ROOT_PATH = Path(__file__).resolve().parent.parent

CONFIG_PATH = ROOT_PATH / 'config'

DATA_PATH = ROOT_PATH / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
FINAL_DATA_PATH = DATA_PATH / "final"

MODELS_PATH = ROOT_PATH / "models"

ARTIFACTS_PATH = ROOT_PATH / "artifacts"

NOTEBOOKS_PATH = ROOT_PATH / "notebooks"

UTILS_PATH = ROOT_PATH / "utils"

SRC_PATH = ROOT_PATH / "src"
