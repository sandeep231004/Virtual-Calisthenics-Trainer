import json
import os
from typing import Any, Dict

def load_pushup_config(config_path: str = None) -> Dict[str, Any]:
    """Load pushup config from JSON file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "pushup_config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def load_squat_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'squat_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)
