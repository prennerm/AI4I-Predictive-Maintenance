"""
Feature Utilities Module for AI4I Predictive Maintenance Project

This module provides utility functions used by both feature_engineering.py and 
feature_selection.py modules. It contains common operations for:
- Feature validation and quality checks
- Safe transformations and data type handling
- Feature naming and organization
- Feature set management and I/O operations
- Integration helpers for pipeline coordination

Author: AI4I Team
Date: August 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import json
import warnings
from scipy import stats
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
import joblib

# Configure logging
logger = logging.getLogger(__name__)

def validate_feature_quality(features: pd.DataFrame, 
                           report_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Perform comprehensive quality validation on feature set.
    
    Args:
        features (pd.DataFrame): Feature matrix to validate
        report_threshold (float): Threshold for reporting issues
        
    Returns:
        Dict[str, Any]: Quality validation report
    """
    logger.info("Validating feature quality")
    
    report = {
        'total_features': len(features.columns),
        'total_samples': len(features),
        'issues_found': [],
        'warnings': [],
        'feature_stats': {}
    }
    
    for col in features.columns:
        feature_stats = {
            'dtype': str(features[col].dtype),
            'missing_count': features[col].isnull().sum(),
            'missing_percentage': features[col].isnull().mean(),
            'unique_values': features[col].nunique(),
            'is_constant': features[col].nunique() <= 1
        }
        
        # Check for issues
        if feature_stats['missing_percentage'] > report_threshold:
            report['warnings'].append(f"High missing values in {col}: {feature_stats['missing_percentage']:.2%}")
        
        if feature_stats['is_constant']:
            report['issues_found'].append(f"Constant feature: {col}")
        
        # Check for infinite values
        if np.isinf(features[col]).any():
            report['issues_found'].append(f"Infinite values in {col}")
        
        # Check for very high cardinality (potential ID columns)
        if feature_stats['unique_values'] > len(features) * 0.9:
            report['warnings'].append(f"Very high cardinality in {col}: {feature_stats['unique_values']} unique values")
        
        report['feature_stats'][col] = feature_stats
    
    report['quality_score'] = 1.0 - len(report['issues_found']) / len(features.columns)
    
    logger.info(f"Quality validation completed. Score: {report['quality_score']:.2f}")
    return report

def check_feature_correlation(features: pd.DataFrame, 
                            threshold: float = 0.95,
                            method: str = 'pearson') -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Identify highly correlated feature pairs.
    
    Args:
        features (pd.DataFrame): Feature matrix
        threshold (float): Correlation threshold
        method (str): Correlation method ('pearson', 'spearman')
        
    Returns:
        Dict: High correlation pairs and recommendations
    """
    logger.info(f"Checking feature correlations (threshold={threshold}, method={method})")
    
    # Calculate correlation matrix
    corr_matrix = features.corr(method=method).abs()
    
    # Find high correlation pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if corr_value > threshold:
                feature1 = corr_matrix.columns[i]
                feature2 = corr_matrix.columns[j]
                high_corr_pairs.append((feature1, feature2, corr_value))
    
    # Generate removal recommendations
    features_to_remove = set()
    for feat1, feat2, corr_val in high_corr_pairs:
        # Keep the feature with higher variance
        if features[feat1].var() >= features[feat2].var():
            features_to_remove.add(feat2)
        else:
            features_to_remove.add(feat1)
    
    result = {
        'high_correlation_pairs': high_corr_pairs,
        'recommended_removals': list(features_to_remove),
        'correlation_matrix': corr_matrix
    }
    
    logger.info(f"Found {len(high_corr_pairs)} high correlation pairs")
    return result

def detect_constant_features(features: pd.DataFrame, 
                           variance_threshold: float = 1e-6) -> List[str]:
    """
    Detect features with zero or near-zero variance.
    
    Args:
        features (pd.DataFrame): Feature matrix
        variance_threshold (float): Minimum variance threshold
        
    Returns:
        List[str]: Names of constant/near-constant features
    """
    logger.info(f"Detecting constant features (variance_threshold={variance_threshold})")
    
    # Calculate variances
    variances = features.var()
    
    # Find constant features
    constant_features = variances[variances <= variance_threshold].index.tolist()
    
    logger.info(f"Found {len(constant_features)} constant features")
    return constant_features

def safe_log_transform(series: pd.Series, 
                      add_constant: float = 1e-6) -> pd.Series:
    """
    Safely apply logarithmic transformation handling zeros and negatives.
    
    Args:
        series (pd.Series): Input series
        add_constant (float): Constant to add before log transformation
        
    Returns:
        pd.Series: Log-transformed series
    """
    # Handle negative values by shifting
    if series.min() <= 0:
        shift_value = abs(series.min()) + add_constant
        shifted_series = series + shift_value
    else:
        shifted_series = series + add_constant
    
    return np.log1p(shifted_series)

def safe_sqrt_transform(series: pd.Series) -> pd.Series:
    """
    Safely apply square root transformation handling negative values.
    
    Args:
        series (pd.Series): Input series
        
    Returns:
        pd.Series: Square root transformed series
    """
    # For negative values, use signed square root
    return np.sign(series) * np.sqrt(np.abs(series))

def robust_scaler_transform(features: pd.DataFrame,
                          quantile_range: Tuple[float, float] = (25.0, 75.0)) -> pd.DataFrame:
    """
    Apply robust scaling using quantiles (outlier-resistant).
    
    Args:
        features (pd.DataFrame): Feature matrix
        quantile_range (Tuple): Quantile range for scaling
        
    Returns:
        pd.DataFrame: Robust scaled features
    """
    logger.info(f"Applying robust scaling (quantile_range={quantile_range})")
    
    scaler = RobustScaler(quantile_range=quantile_range)
    scaled_features = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index
    )
    
    return scaled_features

def create_polynomial_features(features: pd.DataFrame,
                             degree: int = 2,
                             interaction_only: bool = False,
                             max_features: int = 1000) -> pd.DataFrame:
    """
    Create polynomial features with overflow protection.
    
    Args:
        features (pd.DataFrame): Input features
        degree (int): Polynomial degree
        interaction_only (bool): Only interaction terms, no powers
        max_features (int): Maximum number of output features
        
    Returns:
        pd.DataFrame: Polynomial features
    """
    logger.info(f"Creating polynomial features (degree={degree}, interaction_only={interaction_only})")
    
    # Limit input features to prevent explosion
    if len(features.columns) > 10:
        logger.warning(f"Too many features for polynomial expansion. Using top 10 by variance.")
        top_features = features.var().nlargest(10).index
        features_subset = features[top_features]
    else:
        features_subset = features
    
    poly = PolynomialFeatures(
        degree=degree, 
        interaction_only=interaction_only,
        include_bias=False
    )
    
    poly_features = poly.fit_transform(features_subset)
    
    # Limit output size
    if poly_features.shape[1] > max_features:
        logger.warning(f"Polynomial features exceed max_features. Truncating to {max_features}")
        poly_features = poly_features[:, :max_features]
    
    # Create feature names
    feature_names = poly.get_feature_names_out(features_subset.columns)[:poly_features.shape[1]]
    
    poly_df = pd.DataFrame(
        poly_features,
        columns=feature_names,
        index=features.index
    )
    
    logger.info(f"Created {poly_df.shape[1]} polynomial features")
    return poly_df

def generate_feature_names(base_features: List[str], 
                         operations: List[str],
                         separator: str = '_') -> List[str]:
    """
    Generate systematic feature names from base features and operations.
    
    Args:
        base_features (List[str]): Base feature names
        operations (List[str]): Operations applied
        separator (str): Separator character
        
    Returns:
        List[str]: Generated feature names
    """
    feature_names = []
    
    for base_feature in base_features:
        for operation in operations:
            # Clean base feature name (remove special characters)
            clean_base = base_feature.replace('[', '').replace(']', '').replace(' ', '_')
            feature_name = f"{clean_base}{separator}{operation}"
            feature_names.append(feature_name)
    
    return feature_names

def categorize_features(features: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize features by their type/domain based on naming patterns.
    
    Args:
        features (pd.DataFrame): Feature matrix
        
    Returns:
        Dict[str, List[str]]: Feature categories
    """
    logger.info("Categorizing features by type")
    
    categories = {
        'temperature': [],
        'mechanical': [],
        'wear': [],
        'power': [],
        'operational': [],
        'interaction': [],
        'statistical': [],
        'encoded': [],
        'other': []
    }
    
    for feature in features.columns:
        feature_lower = feature.lower()
        
        if any(temp_word in feature_lower for temp_word in ['temp', 'temperature']):
            categories['temperature'].append(feature)
        elif any(mech_word in feature_lower for mech_word in ['speed', 'rpm', 'torque', 'rotational']):
            categories['mechanical'].append(feature)
        elif any(wear_word in feature_lower for wear_word in ['wear', 'tool']):
            categories['wear'].append(feature)
        elif any(power_word in feature_lower for power_word in ['power', 'mechanical_power', 'energy']):
            categories['power'].append(feature)
        elif any(op_word in feature_lower for op_word in ['stress', 'efficiency', 'operational']):
            categories['operational'].append(feature)
        elif any(int_word in feature_lower for int_word in ['interaction', '_x_', 'ratio', 'product']):
            categories['interaction'].append(feature)
        elif any(stat_word in feature_lower for stat_word in ['squared', 'cubed', 'log', 'sqrt', 'zscore']):
            categories['statistical'].append(feature)
        elif any(enc_word in feature_lower for enc_word in ['type_', '_low', '_medium', '_high']):
            categories['encoded'].append(feature)
        else:
            categories['other'].append(feature)
    
    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v}
    
    logger.info(f"Categorized features into {len(categories)} categories")
    return categories

def create_feature_metadata(features: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive metadata table for features.
    
    Args:
        features (pd.DataFrame): Feature matrix
        
    Returns:
        pd.DataFrame: Feature metadata
    """
    logger.info("Creating feature metadata")
    
    categories = categorize_features(features)
    
    # Create reverse mapping
    feature_to_category = {}
    for category, feature_list in categories.items():
        for feature in feature_list:
            feature_to_category[feature] = category
    
    metadata = []
    for feature in features.columns:
        feature_stats = {
            'feature_name': feature,
            'category': feature_to_category.get(feature, 'other'),
            'dtype': str(features[feature].dtype),
            'missing_count': features[feature].isnull().sum(),
            'missing_percentage': features[feature].isnull().mean(),
            'unique_values': features[feature].nunique(),
            'min_value': features[feature].min() if pd.api.types.is_numeric_dtype(features[feature]) else None,
            'max_value': features[feature].max() if pd.api.types.is_numeric_dtype(features[feature]) else None,
            'mean_value': features[feature].mean() if pd.api.types.is_numeric_dtype(features[feature]) else None,
            'std_value': features[feature].std() if pd.api.types.is_numeric_dtype(features[feature]) else None,
            'variance': features[feature].var() if pd.api.types.is_numeric_dtype(features[feature]) else None,
            'skewness': features[feature].skew() if pd.api.types.is_numeric_dtype(features[feature]) else None,
            'kurtosis': features[feature].kurtosis() if pd.api.types.is_numeric_dtype(features[feature]) else None
        }
        metadata.append(feature_stats)
    
    metadata_df = pd.DataFrame(metadata)
    logger.info(f"Created metadata for {len(metadata_df)} features")
    
    return metadata_df

def ensure_numeric_types(features: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all features are numeric, converting where possible.
    
    Args:
        features (pd.DataFrame): Feature matrix
        
    Returns:
        pd.DataFrame: Feature matrix with numeric types
    """
    logger.info("Ensuring numeric data types")
    
    numeric_features = features.copy()
    
    for col in features.columns:
        if not pd.api.types.is_numeric_dtype(features[col]):
            try:
                numeric_features[col] = pd.to_numeric(features[col], errors='coerce')
                logger.info(f"Converted {col} to numeric")
            except:
                logger.warning(f"Could not convert {col} to numeric, dropping")
                numeric_features = numeric_features.drop(columns=[col])
    
    return numeric_features

def handle_missing_in_features(features: pd.DataFrame, 
                             strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in feature sets with appropriate strategies.
    
    Args:
        features (pd.DataFrame): Feature matrix
        strategy (str): Strategy ('median', 'mean', 'mode', 'drop', 'interpolate')
        
    Returns:
        pd.DataFrame: Feature matrix with handled missing values
    """
    logger.info(f"Handling missing values in features using {strategy} strategy")
    
    if strategy == 'median':
        return features.fillna(features.median())
    elif strategy == 'mean':
        return features.fillna(features.mean())
    elif strategy == 'mode':
        return features.fillna(features.mode().iloc[0])
    elif strategy == 'drop':
        return features.dropna()
    elif strategy == 'interpolate':
        return features.interpolate(method='linear')
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def clip_feature_outliers(features: pd.DataFrame, 
                        method: str = 'iqr',
                        factor: float = 1.5) -> pd.DataFrame:
    """
    Clip outliers in feature sets using specified method.
    
    Args:
        features (pd.DataFrame): Feature matrix
        method (str): Method ('iqr', 'zscore', 'percentile')
        factor (float): Clipping factor
        
    Returns:
        pd.DataFrame: Feature matrix with clipped outliers
    """
    logger.info(f"Clipping outliers using {method} method (factor={factor})")
    
    clipped_features = features.copy()
    
    for col in features.select_dtypes(include=[np.number]).columns:
        if method == 'iqr':
            Q1 = features[col].quantile(0.25)
            Q3 = features[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
        elif method == 'zscore':
            mean = features[col].mean()
            std = features[col].std()
            lower_bound = mean - factor * std
            upper_bound = mean + factor * std
            
        elif method == 'percentile':
            lower_percentile = factor
            upper_percentile = 100 - factor
            lower_bound = features[col].quantile(lower_percentile / 100)
            upper_bound = features[col].quantile(upper_percentile / 100)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        clipped_features[col] = features[col].clip(lower_bound, upper_bound)
    
    return clipped_features

def save_feature_set(features: pd.DataFrame, 
                    name: str, 
                    output_dir: str,
                    include_metadata: bool = True) -> None:
    """
    Save feature set with optional metadata.
    
    Args:
        features (pd.DataFrame): Feature set to save
        name (str): Name for the feature set
        output_dir (str): Output directory
        include_metadata (bool): Whether to save metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save feature set
    features_file = output_path / f"{name}_features.csv"
    features.to_csv(features_file, index=False)
    
    if include_metadata:
        # Save metadata
        metadata = create_feature_metadata(features)
        metadata_file = output_path / f"{name}_metadata.csv"
        metadata.to_csv(metadata_file, index=False)
        
        # Save quality report
        quality_report = validate_feature_quality(features)
        quality_file = output_path / f"{name}_quality_report.json"
        with open(quality_file, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
    
    logger.info(f"Feature set '{name}' saved to {output_dir}")

def load_feature_set(name: str, input_dir: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load feature set with optional metadata.
    
    Args:
        name (str): Name of the feature set
        input_dir (str): Input directory
        
    Returns:
        Tuple: (features, metadata)
    """
    input_path = Path(input_dir)
    
    # Load features
    features_file = input_path / f"{name}_features.csv"
    if not features_file.exists():
        raise FileNotFoundError(f"Feature set not found: {features_file}")
    
    features = pd.read_csv(features_file)
    
    # Load metadata if available
    metadata_file = input_path / f"{name}_metadata.csv"
    metadata = None
    if metadata_file.exists():
        metadata = pd.read_csv(metadata_file)
    
    logger.info(f"Loaded feature set '{name}' from {input_dir}")
    return features, metadata

def compare_feature_sets(set1: pd.DataFrame, 
                        set2: pd.DataFrame,
                        name1: str = "Set1",
                        name2: str = "Set2") -> Dict[str, Any]:
    """
    Compare two feature sets and provide detailed comparison.
    
    Args:
        set1 (pd.DataFrame): First feature set
        set2 (pd.DataFrame): Second feature set
        name1 (str): Name of first set
        name2 (str): Name of second set
        
    Returns:
        Dict: Comparison results
    """
    logger.info(f"Comparing feature sets: {name1} vs {name2}")
    
    # Basic statistics
    comparison = {
        'set_names': [name1, name2],
        'feature_counts': [len(set1.columns), len(set2.columns)],
        'sample_counts': [len(set1), len(set2)],
        'common_features': list(set(set1.columns) & set(set2.columns)),
        'unique_to_set1': list(set(set1.columns) - set(set2.columns)),
        'unique_to_set2': list(set(set2.columns) - set(set1.columns)),
        'overlap_percentage': len(set(set1.columns) & set(set2.columns)) / len(set(set1.columns) | set(set2.columns))
    }
    
    # Category comparison
    cat1 = categorize_features(set1)
    cat2 = categorize_features(set2)
    
    comparison['category_comparison'] = {
        name1: {k: len(v) for k, v in cat1.items()},
        name2: {k: len(v) for k, v in cat2.items()}
    }
    
    logger.info(f"Comparison completed. Overlap: {comparison['overlap_percentage']:.2%}")
    return comparison

def merge_feature_sets(baseline: pd.DataFrame, 
                      engineered: pd.DataFrame,
                      handle_duplicates: str = 'drop') -> pd.DataFrame:
    """
    Intelligently merge different feature sets.
    
    Args:
        baseline (pd.DataFrame): Baseline features
        engineered (pd.DataFrame): Engineered features
        handle_duplicates (str): How to handle duplicates ('drop', 'suffix', 'keep_first')
        
    Returns:
        pd.DataFrame: Merged feature set
    """
    logger.info("Merging feature sets")
    
    if handle_duplicates == 'drop':
        # Remove common columns from engineered set
        common_cols = set(baseline.columns) & set(engineered.columns)
        if common_cols:
            logger.info(f"Dropping {len(common_cols)} duplicate columns from engineered set")
            engineered_clean = engineered.drop(columns=list(common_cols))
        else:
            engineered_clean = engineered
        
        merged = pd.concat([baseline, engineered_clean], axis=1)
        
    elif handle_duplicates == 'suffix':
        # Add suffixes to duplicate columns
        merged = pd.concat([baseline, engineered], axis=1, 
                          keys=['baseline', 'engineered'])
        merged.columns = ['_'.join(col).strip() for col in merged.columns.values]
        
    elif handle_duplicates == 'keep_first':
        # Keep baseline version of duplicates
        merged = baseline.copy()
        unique_engineered = engineered.drop(columns=[col for col in engineered.columns 
                                                   if col in baseline.columns])
        merged = pd.concat([merged, unique_engineered], axis=1)
        
    else:
        raise ValueError(f"Unknown duplicate handling strategy: {handle_duplicates}")
    
    logger.info(f"Merged feature sets: {len(merged.columns)} total features")
    return merged

def align_features_with_target(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Ensure features and target are properly aligned.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        
    Returns:
        Tuple: Aligned (X, y)
    """
    # Check for index alignment
    if not X.index.equals(y.index):
        logger.warning("Feature and target indices not aligned. Aligning by intersection.")
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
    else:
        X_aligned = X
        y_aligned = y
    
    # Check for length mismatch
    if len(X_aligned) != len(y_aligned):
        min_length = min(len(X_aligned), len(y_aligned))
        logger.warning(f"Length mismatch. Truncating to {min_length} samples.")
        X_aligned = X_aligned.iloc[:min_length]
        y_aligned = y_aligned.iloc[:min_length]
    
    logger.info(f"Aligned features and target: {X_aligned.shape[0]} samples, {X_aligned.shape[1]} features")
    return X_aligned, y_aligned

def prepare_features_for_model(features: pd.DataFrame, 
                             model_type: str,
                             target: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Prepare features for specific model types.
    
    Args:
        features (pd.DataFrame): Feature matrix
        model_type (str): Type of model ('tree', 'linear', 'neural', 'ensemble')
        target (pd.Series): Target variable (optional)
        
    Returns:
        pd.DataFrame: Prepared feature matrix
    """
    logger.info(f"Preparing features for {model_type} model")
    
    prepared_features = features.copy()
    
    if model_type == 'linear':
        # Linear models need scaled features and no multicollinearity
        corr_info = check_feature_correlation(prepared_features, threshold=0.9)
        if corr_info['recommended_removals']:
            logger.info(f"Removing {len(corr_info['recommended_removals'])} correlated features for linear model")
            prepared_features = prepared_features.drop(columns=corr_info['recommended_removals'])
        
        # Scale features
        prepared_features = robust_scaler_transform(prepared_features)
        
    elif model_type == 'tree' or model_type == 'ensemble':
        # Tree models handle features well, just ensure numeric
        prepared_features = ensure_numeric_types(prepared_features)
        
    elif model_type == 'neural':
        # Neural networks need scaled features and no constant features
        constant_features = detect_constant_features(prepared_features)
        if constant_features:
            logger.info(f"Removing {len(constant_features)} constant features for neural model")
            prepared_features = prepared_features.drop(columns=constant_features)
        
        # Scale features
        prepared_features = robust_scaler_transform(prepared_features)
        
    else:
        logger.warning(f"Unknown model type: {model_type}. Applying basic preparation.")
        prepared_features = ensure_numeric_types(prepared_features)
    
    logger.info(f"Prepared {prepared_features.shape[1]} features for {model_type} model")
    return prepared_features


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing feature utilities...")
    
    # Create sample data for testing
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Air_temperature_K': np.random.normal(300, 5, 1000),
        'Process_temperature_K': np.random.normal(310, 8, 1000),
        'Rotational_speed_rpm': np.random.normal(1500, 200, 1000),
        'Torque_Nm': np.random.normal(40, 10, 1000),
        'Tool_wear_min': np.random.exponential(50, 1000),
        'temp_difference': np.random.normal(10, 2, 1000),  # Correlated with above
        'power_calculated': np.random.normal(5000, 1000, 1000),
        'constant_feature': np.ones(1000),  # Constant feature
    })
    
    # Test utility functions
    print("\n=== Quality Validation ===")
    quality_report = validate_feature_quality(sample_data)
    print(f"Quality Score: {quality_report['quality_score']:.2f}")
    print(f"Issues Found: {len(quality_report['issues_found'])}")
    
    print("\n=== Feature Categorization ===")
    categories = categorize_features(sample_data)
    for category, features in categories.items():
        print(f"{category}: {len(features)} features")
    
    print("\n=== Correlation Check ===")
    corr_info = check_feature_correlation(sample_data, threshold=0.7)
    print(f"High correlation pairs: {len(corr_info['high_correlation_pairs'])}")
    
    print("\n=== Constant Features ===")
    constant_features = detect_constant_features(sample_data)
    print(f"Constant features: {constant_features}")
    
    logger.info("Feature utilities testing completed!")
