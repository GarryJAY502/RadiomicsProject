import yaml
import os

def load_radiomics_config(config_path: str):
    """
    Loads the YAML config for PyRadiomics.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
