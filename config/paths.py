from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent

CONFIG_PATH         = ROOT_PATH / "config"
RAW_DATA_PATH       = ROOT_PATH / "data" / "raw"
PROCESSED_DATA_PATH = ROOT_PATH / "data" / "processed"
MODELS_PATH         = ROOT_PATH / "models"
ARTIFACTS_PATH      = ROOT_PATH / "artifacts"   # preprocessing contract (valid_users/movies)
OUTPUTS_PATH        = ROOT_PATH / "outputs"     # pipeline outputs (test set, recommendations)
