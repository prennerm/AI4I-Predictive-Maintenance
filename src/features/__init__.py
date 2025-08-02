"""
Features module for AI4I Predictive Maintenance Project

Contains feature engineering, selection, and utility functions.
"""

from .feature_engineering import FeatureEngineer, create_feature_sets_for_comparison
from .feature_selection import FeatureSelector, create_comparison_feature_sets
from .feature_utils import (
    validate_feature_quality, check_feature_correlation, detect_constant_features,
    safe_log_transform, safe_sqrt_transform, robust_scaler_transform,
    create_polynomial_features, generate_feature_names, categorize_features,
    create_feature_metadata, ensure_numeric_types, handle_missing_in_features,
    clip_feature_outliers, save_feature_set, load_feature_set, compare_feature_sets,
    merge_feature_sets, align_features_with_target, prepare_features_for_model
)

__all__ = [
    'FeatureEngineer', 'FeatureSelector',
    'create_feature_sets_for_comparison', 'create_comparison_feature_sets',
    'validate_feature_quality', 'check_feature_correlation', 'detect_constant_features',
    'safe_log_transform', 'safe_sqrt_transform', 'robust_scaler_transform',
    'create_polynomial_features', 'generate_feature_names', 'categorize_features',
    'create_feature_metadata', 'ensure_numeric_types', 'handle_missing_in_features',
    'clip_feature_outliers', 'save_feature_set', 'load_feature_set', 'compare_feature_sets',
    'merge_feature_sets', 'align_features_with_target', 'prepare_features_for_model'
]
