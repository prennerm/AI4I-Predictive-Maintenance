"""
Data Preprocessing Module for AI4I Predictive Maintenance Project

This module provides comprehensive data preprocessing functionality including:
- Data loading and validation
- Missing value handling
- Outlier detection (reporting only, no modification)
- Data normalization and scaling
- Feature encoding and transformation
- Data quality assessment

Author: AI4I Team
Date: August 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing class for the AI4I dataset.
    
    This class handles all data preprocessing steps from raw data loading
    to final prepared datasets ready for machine learning models.
    """
    
    def __init__(self, data_path: str, random_state: int = 42):
        """
        Initialize the DataPreprocessor.
        
        Args:
            data_path (str): Path to the raw data file
            random_state (int): Random state for reproducibility
        """
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.scaler = None
        self.label_encoders = {}
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = None
        self.target_column = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the raw dataset from CSV file.
        
        Returns:
            pd.DataFrame: Loaded raw dataset
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the data file is empty or corrupted
        """
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
                
            logger.info(f"Loading data from {self.data_path}")
            self.raw_data = pd.read_csv(self.data_path)
            
            if self.raw_data.empty:
                raise ValueError("Loaded dataset is empty")
                
            logger.info(f"Data loaded successfully. Shape: {self.raw_data.shape}")
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment.
        
        Returns:
            Dict[str, Any]: Data quality report
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        logger.info("Performing data quality assessment...")
        
        quality_report = {
            'shape': self.raw_data.shape,
            'columns': list(self.raw_data.columns),
            'data_types': self.raw_data.dtypes.to_dict(),
            'missing_values': self.raw_data.isnull().sum().to_dict(),
            'missing_percentage': (self.raw_data.isnull().sum() / len(self.raw_data) * 100).to_dict(),
            'duplicates': self.raw_data.duplicated().sum(),
            'numeric_columns': self.raw_data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.raw_data.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Statistical summary for numeric columns
        if quality_report['numeric_columns']:
            quality_report['numeric_summary'] = self.raw_data[quality_report['numeric_columns']].describe().to_dict()
        
        # Unique values for categorical columns
        if quality_report['categorical_columns']:
            quality_report['categorical_summary'] = {
                col: self.raw_data[col].value_counts().to_dict() 
                for col in quality_report['categorical_columns']
            }
        
        logger.info("Data quality assessment completed")
        return quality_report
    
    def handle_missing_values(self, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Strategy for handling missing values
                          'auto', 'drop', 'mean', 'median', 'mode', 'forward_fill'
        
        Returns:
            pd.DataFrame: Dataset with handled missing values
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        data = self.raw_data.copy()
        missing_info = data.isnull().sum()
        
        if missing_info.sum() == 0:
            logger.info("No missing values found in the dataset")
            return data
            
        logger.info(f"Handling missing values using strategy: {strategy}")
        
        if strategy == 'auto':
            # Automatic strategy based on missing percentage
            for col in data.columns:
                missing_pct = (data[col].isnull().sum() / len(data)) * 100
                
                if missing_pct > 50:
                    # Drop columns with >50% missing values
                    data = data.drop(columns=[col])
                    logger.warning(f"Dropped column {col} (>{missing_pct:.1f}% missing)")
                elif missing_pct > 0:
                    if data[col].dtype in ['int64', 'float64']:
                        # Use median for numeric columns
                        data[col] = data[col].fillna(data[col].median())
                    else:
                        # Use mode for categorical columns
                        data[col] = data[col].fillna(data[col].mode()[0])
                        
        elif strategy == 'drop':
            data = data.dropna()
            
        elif strategy in ['mean', 'median']:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if strategy == 'mean':
                    data[col] = data[col].fillna(data[col].mean())
                else:
                    data[col] = data[col].fillna(data[col].median())
                    
        elif strategy == 'mode':
            for col in data.columns:
                if data[col].isnull().sum() > 0:
                    data[col] = data[col].fillna(data[col].mode()[0])
                    
        elif strategy == 'forward_fill':
            data = data.fillna(method='ffill')
            
        logger.info(f"Missing values handled. New shape: {data.shape}")
        return data
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers in numeric columns without modifying the data.
        
        Args:
            data (pd.DataFrame): Input dataset
            method (str): Method for outlier detection ('iqr', 'zscore')
            threshold (float): Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Original dataset (unchanged)
        """
        logger.info(f"Detecting outliers using {method} method")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = z_scores > threshold
                
            outlier_count = outliers.sum()
            outlier_info[col] = outlier_count
            
            if outlier_count > 0:
                logger.info(f"Detected {outlier_count} outliers in column {col} (no modification applied)")
        
        total_outliers = sum(outlier_info.values())
        logger.info(f"Total outliers detected: {total_outliers} (data unchanged)")
        
        return data
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using appropriate encoding methods.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical features
        """
        logger.info("Encoding categorical features")
        
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            logger.info("No categorical columns found")
            return data
            
        encoded_data = data.copy()
        
        for col in categorical_cols:
            unique_values = data[col].nunique()
            
            if unique_values == 2:
                # Binary encoding for binary categorical variables
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
                logger.info(f"Binary encoded column: {col}")
                
            elif unique_values <= 10:
                # One-hot encoding for low cardinality categorical variables
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                encoded_data = pd.concat([encoded_data.drop(columns=[col]), dummies], axis=1)
                logger.info(f"One-hot encoded column: {col}")
                
            else:
                # Label encoding for high cardinality categorical variables
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
                logger.info(f"Label encoded column: {col}")
        
        return encoded_data
    
    def scale_features(self, data: pd.DataFrame, scaler_type: str = 'standard',
                      exclude_columns: Optional[list] = None) -> pd.DataFrame:
        """
        Scale numeric features using specified scaling method.
        
        Args:
            data (pd.DataFrame): Input dataset
            scaler_type (str): Type of scaler ('standard', 'minmax')
            exclude_columns (list): Columns to exclude from scaling
            
        Returns:
            pd.DataFrame: Dataset with scaled features
        """
        logger.info(f"Scaling features using {scaler_type} scaler")
        
        if exclude_columns is None:
            exclude_columns = []
            
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_cols if col not in exclude_columns]
        
        if len(cols_to_scale) == 0:
            logger.info("No columns to scale")
            return data
            
        scaled_data = data.copy()
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
            
        scaled_data[cols_to_scale] = self.scaler.fit_transform(data[cols_to_scale])
        
        logger.info(f"Scaled {len(cols_to_scale)} columns")
        return scaled_data
    
    def prepare_features_target(self, data: pd.DataFrame, target_column: str,
                              feature_columns: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variables for machine learning.
        
        Args:
            data (pd.DataFrame): Processed dataset
            target_column (str): Name of the target column
            feature_columns (list): List of feature columns (if None, use all except target)
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
            
        self.target_column = target_column
        
        if feature_columns is None:
            self.feature_columns = [col for col in data.columns if col != target_column]
        else:
            self.feature_columns = feature_columns
            
        X = data[self.feature_columns]
        y = data[target_column]
        
        logger.info(f"Prepared features: {X.shape[1]} columns, Target: {target_column}")
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                   val_size: float = 0.1, stratify: bool = True) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion of test set
            val_size (float): Proportion of validation set
            stratify (bool): Whether to stratify the split
            
        Returns:
            Tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        stratify_param = y if stratify else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=stratify_param
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        stratify_param_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=stratify_param_temp
        )
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, data: pd.DataFrame, output_dir: str, filename: str = "processed_data.csv"):
        """
        Save processed data to specified directory.
        
        Args:
            data (pd.DataFrame): Processed dataset
            output_dir (str): Output directory path
            filename (str): Output filename
        """
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to: {output_path}")
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all preprocessing steps performed.
        
        Returns:
            Dict[str, Any]: Preprocessing summary
        """
        summary = {
            'data_shape': self.processed_data.shape if self.processed_data is not None else None,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'label_encoders': list(self.label_encoders.keys()),
            'preprocessing_steps_completed': [
                'data_loading',
                'quality_assessment',
                'missing_value_handling',
                'outlier_detection_only',
                'categorical_encoding',
                'feature_scaling'
            ]
        }
        
        return summary


def preprocess_ai4i_data(data_path: str, output_dir: str, 
                         target_column: str = 'Machine failure') -> Dict[str, Any]:
    """
    Main preprocessing pipeline for AI4I dataset.
    
    Args:
        data_path (str): Path to raw data file
        output_dir (str): Directory to save processed data
        target_column (str): Name of target column
        
    Returns:
        Dict[str, Any]: Preprocessing results and summary
    """
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor(data_path)
        
        # Load and validate data
        raw_data = preprocessor.load_data()
        quality_report = preprocessor.validate_data_quality()
        
        # Preprocessing pipeline
        data = preprocessor.handle_missing_values(strategy='auto')
        data = preprocessor.detect_outliers(data, method='iqr')
        data = preprocessor.encode_categorical_features(data)
        data = preprocessor.scale_features(data, scaler_type='standard', 
                                         exclude_columns=[target_column])
        
        # Save processed data
        preprocessor.save_processed_data(data, output_dir)
        preprocessor.processed_data = data
        
        # Prepare features and target
        X, y = preprocessor.prepare_features_target(data, target_column)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        # Save split datasets
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        preprocessor.save_processed_data(train_data, output_dir, "train_data.csv")
        preprocessor.save_processed_data(val_data, output_dir, "val_data.csv")
        preprocessor.save_processed_data(test_data, output_dir, "test_data.csv")
        
        # Generate summary
        summary = preprocessor.get_preprocessing_summary()
        summary['quality_report'] = quality_report
        summary['data_splits'] = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }
        
        logger.info("Data preprocessing pipeline completed successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    data_path = "data/raw/ai4i2020.csv"
    output_dir = "data/processed"
    
    summary = preprocess_ai4i_data(data_path, output_dir)
    print("Preprocessing Summary:", summary)
