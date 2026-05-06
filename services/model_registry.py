import json
from pathlib import Path

ACTIVE_MODEL_FILE = Path("models/active_model.json")


def get_active_model_config(model_key: str):
    with open(ACTIVE_MODEL_FILE, "r") as f:
        config = json.load(f)

    if model_key not in config:
        raise ValueError(f"No active model config found for {model_key}")

    return config[model_key]