"""
Tests for config module.

Covers:
  - _DEFAULT_CONFIG structure completeness
  - load_config() with default path (no YAML file present)
  - load_config() loading from a real YAML file
  - Deep merge logic (nested keys merged, not replaced)
  - Partial YAML override (only some keys provided)
  - CLI-style override simulation
  - Missing config file falls back to defaults
  - PyYAML not installed scenario
"""

import os
import sys
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from src.config import load_config, _DEFAULT_CONFIG


class TestDefaultConfigStructure:
    """Test _DEFAULT_CONFIG dictionary structure."""

    def test_top_level_keys(self):
        expected = {'acquisition', 'eeg_stream', 'bci', 'decoder', 'llm', 'feedback'}
        assert set(_DEFAULT_CONFIG.keys()) == expected

    def test_acquisition_section(self):
        acq = _DEFAULT_CONFIG['acquisition']
        assert acq['sample_rate'] == 250
        assert acq['window_size'] == 1000
        assert acq['window_overlap'] == 500

    def test_eeg_stream_section(self):
        es = _DEFAULT_CONFIG['eeg_stream']
        assert es['window_seconds'] == 2.0
        assert es['push_interval'] == 0.2
        assert es['display_channels'] == 8
        assert es['sample_rate'] == 250

    def test_bci_section(self):
        bci = _DEFAULT_CONFIG['bci']
        assert bci['debounce_frames'] == 3
        assert bci['confidence_threshold'] == 0.5
        assert bci['selection_timeout'] == 15.0

    def test_decoder_section(self):
        dec = _DEFAULT_CONFIG['decoder']
        assert dec['model_sample_rate'] == 128.0
        assert dec['bandpass_low'] == 4.0
        assert dec['bandpass_high'] == 38.0
        assert dec['n_classes'] == 4
        assert dec['class_labels'] == ['left_hand', 'right_hand', 'feet', 'rest']

    def test_llm_section_nested(self):
        llm = _DEFAULT_CONFIG['llm']
        assert llm['default_backend'] == 'mock'
        assert llm['ollama']['model'] == 'qwen2.5:7b'
        assert llm['ollama']['host'] == 'http://localhost:11434'
        assert llm['api']['url'] == ''
        assert llm['api']['key'] == ''
        assert llm['api']['model'] == 'gpt-4o-mini'

    def test_feedback_section(self):
        fb = _DEFAULT_CONFIG['feedback']
        assert fb['host'] == '127.0.0.1'
        assert fb['port'] == 8080


class TestLoadConfigDefaults:
    """Test load_config() when no YAML file is available."""

    def test_returns_dict(self):
        config = load_config('/nonexistent/path.yaml')
        assert isinstance(config, dict)

    def test_missing_file_returns_defaults(self):
        config = load_config('/nonexistent/path.yaml')
        assert config == _DEFAULT_CONFIG

    def test_defaults_are_deep_copy(self):
        """Ensure returned config is not the same object as _DEFAULT_CONFIG."""
        config = load_config('/nonexistent/path.yaml')
        assert config is not _DEFAULT_CONFIG
        config['acquisition']['sample_rate'] = 999
        assert _DEFAULT_CONFIG['acquisition']['sample_rate'] == 250


class TestLoadConfigFromYaml:
    """Test load_config() reading from an actual YAML file."""

    @pytest.fixture
    def yaml_file(self, tmp_path):
        """Create a temporary YAML config file."""
        content = (
            "acquisition:\n"
            "  sample_rate: 500\n"
            "eeg_stream:\n"
            "  window_seconds: 3.0\n"
            "  push_interval: 0.5\n"
            "llm:\n"
            "  ollama:\n"
            "    model: llama3:8b\n"
        )
        fpath = tmp_path / "test_config.yaml"
        fpath.write_text(content, encoding='utf-8')
        return str(fpath)

    def test_loads_top_level_override(self, yaml_file):
        config = load_config(yaml_file)
        assert config['acquisition']['sample_rate'] == 500

    def test_deep_merge_preserves_unoverridden_keys(self, yaml_file):
        """Keys not in YAML should retain default values."""
        config = load_config(yaml_file)
        # window_size and window_overlap not in YAML → keep defaults
        assert config['acquisition']['window_size'] == 1000
        assert config['acquisition']['window_overlap'] == 500

    def test_deep_merge_nested_dict(self, yaml_file):
        """Nested dicts are merged, not replaced."""
        config = load_config(yaml_file)
        # ollama.model overridden, ollama.host preserved
        assert config['llm']['ollama']['model'] == 'llama3:8b'
        assert config['llm']['ollama']['host'] == 'http://localhost:11434'
        # api sub-dict fully preserved
        assert config['llm']['api']['model'] == 'gpt-4o-mini'

    def test_deep_merge_multiple_sections(self, yaml_file):
        config = load_config(yaml_file)
        assert config['eeg_stream']['window_seconds'] == 3.0
        assert config['eeg_stream']['push_interval'] == 0.5
        # display_channels not in YAML → default
        assert config['eeg_stream']['display_channels'] == 8

    def test_sections_not_in_yaml_preserved(self, yaml_file):
        """Sections absent from YAML file remain at defaults."""
        config = load_config(yaml_file)
        assert config['bci']['debounce_frames'] == 3
        assert config['decoder']['n_classes'] == 4
        assert config['feedback']['port'] == 8080


class TestLoadConfigEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_yaml_file(self, tmp_path):
        fpath = tmp_path / "empty.yaml"
        fpath.write_text("", encoding='utf-8')
        config = load_config(str(fpath))
        assert config == _DEFAULT_CONFIG

    def test_yaml_with_null_content(self, tmp_path):
        fpath = tmp_path / "null.yaml"
        fpath.write_text("---\n", encoding='utf-8')
        config = load_config(str(fpath))
        assert config == _DEFAULT_CONFIG

    def test_pyyaml_not_installed(self, tmp_path):
        """When PyYAML is not installed, should fall back to defaults."""
        fpath = tmp_path / "config.yaml"
        fpath.write_text("acquisition:\n  sample_rate: 500\n", encoding='utf-8')
        with patch('src.config.yaml', None):
            config = load_config(str(fpath))
        assert config == _DEFAULT_CONFIG

    def test_config_path_none_uses_default_location(self):
        """When config_path is None, should look for configs/phase1.yaml."""
        # This should either load the real file or fall back to defaults
        config = load_config(None)
        assert isinstance(config, dict)
        assert 'acquisition' in config
        assert 'decoder' in config


class TestCliOverrideSimulation:
    """Simulate CLI argument overrides on top of loaded config."""

    def test_override_after_load(self, tmp_path):
        fpath = tmp_path / "config.yaml"
        fpath.write_text("acquisition:\n  sample_rate: 500\n", encoding='utf-8')
        config = load_config(str(fpath))

        # Simulate CLI override
        config['acquisition']['sample_rate'] = 1000
        assert config['acquisition']['sample_rate'] == 1000
        # Other keys untouched
        assert config['acquisition']['window_size'] == 1000

    def test_override_nested_key(self, tmp_path):
        config = load_config('/nonexistent/path.yaml')
        config['llm']['default_backend'] = 'ollama'
        assert config['llm']['default_backend'] == 'ollama'
        # Nested ollama dict preserved
        assert config['llm']['ollama']['model'] == 'qwen2.5:7b'
