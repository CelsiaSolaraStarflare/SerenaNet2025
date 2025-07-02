"""
Configuration utilities for SerenaNet.

This module provides utilities for loading, saving, and managing
configuration files in YAML format.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        config_path (str): Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
    
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")
        raise


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config (Dict[str, Any]): Base configuration
        override_config (Dict[str, Any]): Override configuration
        
    Returns:
        Dict[str, Any]: Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def load_config_with_overrides(
    config_path: str,
    override_config: Optional[Dict[str, Any]] = None,
    override_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load configuration with optional overrides.
    
    Args:
        config_path (str): Path to base configuration file
        override_config (Dict[str, Any], optional): Override configuration dict
        override_file (str, optional): Path to override configuration file
        
    Returns:
        Dict[str, Any]: Final configuration
    """
    # Load base config
    config = load_config(config_path)
    
    # Apply file overrides
    if override_file:
        override_from_file = load_config(override_file)
        config = merge_configs(config, override_from_file)
    
    # Apply dict overrides
    if override_config:
        config = merge_configs(config, override_config)
    
    return config


def resolve_config_paths(config: Dict[str, Any], base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve relative paths in configuration to absolute paths.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        base_dir (str, optional): Base directory for resolving paths
        
    Returns:
        Dict[str, Any]: Configuration with resolved paths
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    base_path = Path(base_dir)
    
    def resolve_paths_recursive(obj):
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key.endswith('_dir') or key.endswith('_path') or key.endswith('_file'):
                    if isinstance(value, str) and not Path(value).is_absolute():
                        result[key] = str(base_path / value)
                    else:
                        result[key] = value
                else:
                    result[key] = resolve_paths_recursive(value)
            return result
        elif isinstance(obj, list):
            return [resolve_paths_recursive(item) for item in obj]
        else:
            return obj
    
    return resolve_paths_recursive(config)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and required fields.
    
    Args:
        config (Dict[str, Any]): Configuration to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = [
        'model',
        'training',
        'data',
        'logging'
    ]
    
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required configuration field: {field}")
            return False
    
    # Validate model config
    model_config = config['model']
    if 'phoneme_vocab_size' not in model_config:
        logger.error("Missing 'phoneme_vocab_size' in model configuration")
        return False
    
    # Validate training config
    training_config = config['training']
    required_training_fields = ['batch_size', 'learning_rate', 'epochs']
    for field in required_training_fields:
        if field not in training_config:
            logger.error(f"Missing required training field: {field}")
            return False
    
    # Validate data config
    data_config = config['data']
    if 'sample_rate' not in data_config:
        logger.error("Missing 'sample_rate' in data configuration")
        return False
    
    logger.info("Configuration validation passed")
    return True


def get_config_summary(config: Dict[str, Any]) -> str:
    """
    Get a summary string of the configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        str: Configuration summary
    """
    summary_lines = []
    
    # Model info
    model_config = config.get('model', {})
    summary_lines.append(f"Model: {model_config.get('name', 'SerenaNet')}")
    summary_lines.append(f"Vocab Size: {model_config.get('phoneme_vocab_size', 41)}")
    
    # Training info
    training_config = config.get('training', {})
    summary_lines.append(f"Batch Size: {training_config.get('batch_size', 8)}")
    summary_lines.append(f"Learning Rate: {training_config.get('learning_rate', 1e-4)}")
    summary_lines.append(f"Epochs: {training_config.get('epochs', 50)}")
    
    # Data info
    data_config = config.get('data', {})
    summary_lines.append(f"Sample Rate: {data_config.get('sample_rate', 16000)}")
    summary_lines.append(f"Mel Bins: {data_config.get('n_mels', 128)}")
    
    return '\n'.join(summary_lines)


class ConfigManager:
    """
    Configuration manager for handling complex configuration scenarios.
    
    Args:
        config_path (str): Path to main configuration file
        config_dir (str, optional): Directory containing configuration files
    """
    
    def __init__(self, config_path: str, config_dir: Optional[str] = None):
        self.config_path = Path(config_path)
        self.config_dir = Path(config_dir) if config_dir else self.config_path.parent
        self.config = self.load()
    
    def load(self) -> Dict[str, Any]:
        """Load configuration with inheritance support."""
        config = load_config(self.config_path)
        
        # Handle defaults inheritance
        if 'defaults' in config:
            base_configs = []
            for default in config['defaults']:
                default_path = self.config_dir / f"{default}.yaml"
                if default_path.exists():
                    base_configs.append(load_config(default_path))
            
            # Merge base configs
            for base_config in base_configs:
                config = merge_configs(base_config, config)
            
            # Remove defaults key
            del config['defaults']
        
        # Resolve paths
        config = resolve_config_paths(config, str(self.config_dir))
        
        # Validate
        if not validate_config(config):
            raise ValueError("Configuration validation failed")
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by dot notation key."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file."""
        save_path = path if path else self.config_path
        save_config(self.config, save_path)
