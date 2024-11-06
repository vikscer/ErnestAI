import yaml
import json
import os

def load_config():
    # Load main config file
    with open("config/config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # Load API keys separately
    api_keys_path = os.path.join("config", "api_keys.json")
    with open(api_keys_path, 'r') as file:
        api_keys = json.load(file)

    # Add API keys to the main config under a separate key
    config['api_keys'] = api_keys
    return config
