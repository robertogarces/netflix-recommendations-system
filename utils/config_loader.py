import yaml
from pathlib import Path

def load_config(config_path: Path) -> dict:
    """Load a YAML file and returns a dictionary"""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
