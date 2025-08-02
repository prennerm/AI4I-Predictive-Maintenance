"""
Configuration management for AI4I Predictive Maintenance Project

Central configuration settings for the entire project.
"""

import os
from pathlib import Path

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
