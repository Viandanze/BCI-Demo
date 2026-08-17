"""
Configuration management utilities
Handles loading, saving, and merging YAML configuration files
"""

import os
import yaml
from typing import Any, Dict, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DictConfig(dict):
    """
    Dictionary-based configuration class with dot notation access support.
    
    Example:
        config = DictConfig({'model': {'lr': 0.001}})
        print(config.model.lr)  # 0.001
    """
    
    def __init__(self, data: Optional[Dict] = None, **kwargs):
        super().__init__()
        if data:
            for key, value in data.items():
                if isinstance(value, dict):
                    self[key] = DictConfig(value)
                else:
                    self[key] = value
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self[key] = DictConfig(value)
            else:
                self[key] = value
    
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, DictConfig):
            self[key] = DictConfig(value)
        else:
            self[key] = value
    
    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{key}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default fallback."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def to_dict(self) -> Dict:
        """Convert configuration back to plain dictionary."""
        result = {}
        for key, value in self.items():
            if isinstance(value, DictConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        DictConfig object with loaded configuration
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML: {e}")
            raise
    
    if config_dict is None:
        config_dict = {}
    
    return DictConfig(config_dict)


def save_config(config: Union[DictConfig, Dict], save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        save_path: Path where to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving configuration to: {save_path}")
    
    if isinstance(config, DictConfig):
        config = config.to_dict()
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """
    Merge two configurations, with override values taking precedence.
    
    Args:
        base_config: Base configuration (lower priority)
        override_config: Override configuration (higher priority)
        
    Returns:
        Merged DictConfig
    """
    result = DictConfig()
    
    # Add all base config values
    for key, value in base_config.items():
        if isinstance(value, DictConfig):
            result[key] = DictConfig(value.to_dict())
        else:
            result[key] = value
    
    # Override with override config values
    for key, value in override_config.items():
        if isinstance(value, DictConfig):
            if key in result and isinstance(result[key], DictConfig):
                # Recursively merge nested configs
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = DictConfig(value.to_dict())
        else:
            result[key] = value
    
    return result


def get_nested_value(config: DictConfig, key_path: str, default: Any = None) -> Any:
    """
    Get value from nested configuration using dot notation.
    
    Args:
        config: Configuration object
        key_path: Dot-separated path (e.g., 'model.hidden_size')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def set_nested_value(config: DictConfig, key_path: str, value: Any) -> None:
    """
    Set value in nested configuration using dot notation.
    
    Args:
        config: Configuration object
        key_path: Dot-separated path (e.g., 'model.hidden_size')
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    for i, key in enumerate(keys[:-1]):
        if key not in current:
            current[key] = DictConfig()
        elif not isinstance(current[key], (dict, DictConfig)):
            raise ValueError(f"Cannot traverse non-dict value at '{'.'.join(keys[:i+1])}'")
        current = current[key]
    
    current[keys[-1]] = value
