import yaml
import os
import sys

_CONFIG_CACHE = None

def load_config(path="config.yaml"):
    """
    Loads the YAML configuration file.
    Caches the result so subsequent calls don't re-read the file.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    # Attempt to find config.yaml in valid locations
    # 1. Absolute path given
    # 2. Current working directory
    # 3. Project root (relative to this file)
    
    candidates = [
        path,
        os.path.join(os.getcwd(), path),
        os.path.join(os.path.dirname(__file__), "..", "..", path)
    ]
    
    config_path = None
    for c in candidates:
        if os.path.exists(c):
            config_path = c
            break
            
    if not config_path:
        # Fallback or error? For now, print warning and return empty dict or raise error
        print(f"Warning: Configuration file '{path}' not found in search paths: {candidates}")
        return {}

    try:
        with open(config_path, 'r') as f:
            _CONFIG_CACHE = yaml.safe_load(f)
        return _CONFIG_CACHE
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}
