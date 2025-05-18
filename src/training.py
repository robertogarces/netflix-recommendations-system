import sys
import logging
import yaml
import mlflow

sys.path.append('../')

from config.paths import CONFIG_PATH
from utils.config_loader import load_config

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import both training modules
from src.ncf_model import train_ncf_model
from src.svd_model import train_svd_model  # Extraemos la lógica SVD actual a esta función (ver abajo)

def main():
    config = load_config(CONFIG_PATH / "settings.yaml")
    model_name = config["model"]["type"].lower()

    mlflow.set_experiment(f"Netflix_{model_name.upper()}_Training")

    logger.info(f"Selected model: {model_name}")
    with mlflow.start_run(run_name=f"{model_name.upper()}_Training_Run"):
        if model_name == "svd":
            train_svd_model(config)
        elif model_name == "ncf":
            train_ncf_model(config)
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'svd' or 'ncf'.")

if __name__ == "__main__":
    main()
