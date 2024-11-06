import yaml
import json
import os

def load_config():
    # Load main config file
    with open("config/config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    character = config['assistant']['character']

    # Load API keys separately
    api_keys_path = os.path.join("config", "api_keys.json")
    with open(api_keys_path, 'r') as file:
        api_keys = json.load(file)

    # Load openai prompt separately
    openai_prompt_path = os.path.join("config", f"openai_prompt_{character}.txt")
    with open(openai_prompt_path, 'r', encoding='utf-8') as file:
        openai_prompt = file.read()

    # Add openai prompt to the main config under a separate key
    config['openai_prompt'] = openai_prompt

    # Add API keys to the main config under a separate key
    config['api_keys'] = api_keys

    return config
