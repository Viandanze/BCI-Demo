"""Configuration management for NeuroDecode Phase 1.

Provides default configuration and YAML-based config loading with deep merge
support. CLI arguments override config file values, which override defaults.
"""

import copy
import logging
import os
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = {
    'acquisition': {
        'sample_rate': 250,
        'window_size': 1000,
        'window_overlap': 500,
    },
    'eeg_stream': {
        'window_seconds': 2.0,
        'push_interval': 0.2,
        'display_channels': 8,
        'sample_rate': 250,
    },
    'bci': {
        'debounce_frames': 3,
        'confidence_threshold': 0.5,
        'selection_timeout': 15.0,
    },
    'decoder': {
        'model_sample_rate': 128.0,
        'bandpass_low': 4.0,
        'bandpass_high': 38.0,
        'n_classes': 4,
        'class_labels': ['left_hand', 'right_hand', 'feet', 'rest'],
    },
    'llm': {
        'default_backend': 'mock',
        'ollama': {'model': 'qwen2.5:7b', 'host': 'http://localhost:11434'},
        'api': {'url': '', 'key': '', 'model': 'gpt-4o-mini'},
    },
    'feedback': {
        'host': '127.0.0.1',
        'port': 8080,
    },
}


def load_config(config_path: Optional[str] = None) -> dict:
    """Load Phase 1 configuration from YAML file.

    Falls back to built-in defaults if file is missing or PyYAML not installed.

    Args:
        config_path: Path to YAML config file. If None, uses configs/phase1.yaml
            relative to the project root.

    Returns:
        Configuration dict with all parameters.
    """
    config = copy.deepcopy(_DEFAULT_CONFIG)

    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs', 'phase1.yaml'
        )

    if yaml is not None and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f) or {}

        def _deep_merge(base, override):
            for k, v in override.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    _deep_merge(base[k], v)
                else:
                    base[k] = v

        _deep_merge(config, loaded)
        logger.info(f"Config loaded from {config_path}")
    else:
        logger.info("Using default config (configs/phase1.yaml not found or PyYAML not installed)")

    return config
