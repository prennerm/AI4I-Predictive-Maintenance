"""
Configuration management for AI4I Predictive Maintenance Project

Central configuration settings for the entire project.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
FEATURES_DATA_DIR = DATA_ROOT / "features"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"

# Default file paths
DEFAULT_RAW_DATA_FILE = RAW_DATA_DIR / "ai4i2020.csv"

# Model configuration
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1

# Feature engineering settings
MAX_FEATURES_FOR_ONEHOT = 10
OUTLIER_THRESHOLD_IQR = 1.5
OUTLIER_THRESHOLD_ZSCORE = 3.0

# Model training settings
CV_FOLDS = 5
N_JOBS = -1

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, FEATURES_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class Config:
    """
    Configuration class for AI4I Predictive Maintenance Project.
    
    Provides centralized configuration management with default values
    and the ability to load from YAML files.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration with defaults and optional YAML override.
        
        Args:
            config_path: Optional path to YAML configuration file
        """
        # Set default configuration
        self._config = self._get_default_config()
        
        # Load from YAML if provided
        if config_path:
            self.load_from_yaml(config_path)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration dictionary."""
        return {
            'data': {
                'raw_data_file': str(DEFAULT_RAW_DATA_FILE),
                'test_size': DEFAULT_TEST_SIZE,
                'val_size': DEFAULT_VAL_SIZE,
                'random_state': DEFAULT_RANDOM_STATE,
                'stratify': True,
                'preprocessing': {
                    'handle_missing': True,
                    'encode_categorical': True,
                    'scale_features': True,
                    'remove_outliers': False,
                    'outlier_threshold_iqr': OUTLIER_THRESHOLD_IQR,
                    'outlier_threshold_zscore': OUTLIER_THRESHOLD_ZSCORE
                }
            },
            'features': {
                'engineering': {
                    'interaction_features': True,
                    'polynomial_features': False,
                    'statistical_features': True,
                    'domain_features': True,
                    'max_features_onehot': MAX_FEATURES_FOR_ONEHOT
                },
                'selection': {
                    'method': 'mutual_info',
                    'max_features': 50,
                    'remove_low_variance': True,
                    'correlation_threshold': 0.95
                }
            },
            'training': {
                'cross_validation': {
                    'cv_folds': CV_FOLDS,
                    'scoring': 'f1_weighted'
                },
                'hyperparameter_optimization': {
                    'enabled': True,
                    'n_trials': 100,
                    'timeout': 3600,  # 1 hour
                    'method': 'random_search'
                },
                'early_stopping': {
                    'enabled': True,
                    'patience': 10
                },
                'models': {
                    'random_forest': True,
                    'xgboost': True,
                    'svm': True,
                    'logistic': True,
                    'mlp': True
                },
                'n_jobs': N_JOBS
            },
            'paths': {
                'project_root': str(PROJECT_ROOT),
                'data_root': str(DATA_ROOT),
                'raw_data_dir': str(RAW_DATA_DIR),
                'processed_data_dir': str(PROCESSED_DATA_DIR),
                'features_data_dir': str(FEATURES_DATA_DIR),
                'models_dir': str(MODELS_DIR),
                'reports_dir': str(REPORTS_DIR)
            },
            'logging': {
                'level': LOG_LEVEL,
                'format': LOG_FORMAT
            }
        }
    
    def load_from_yaml(self, config_path: str) -> None:
        """
        Load configuration from YAML file and merge with defaults.
        
        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Deep merge with defaults
            self._config = self._deep_merge(self._config, yaml_config)
            
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing YAML config {config_path}: {e}. Using defaults.")
    
    def _deep_merge(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override taking precedence.
        
        Args:
            default: Default configuration dictionary
            override: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.test_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.test_size')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def save_to_yaml(self, output_path: str) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save YAML configuration
        """
        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    # Convenience properties for common configurations
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config.get('data', {})
    
    @property
    def features_config(self) -> Dict[str, Any]:
        """Get features configuration."""
        return self._config.get('features', {})
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._config.get('training', {})
    
    @property
    def paths_config(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self._config.get('paths', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get('logging', {})


# Global default configuration instance
default_config = Config()
