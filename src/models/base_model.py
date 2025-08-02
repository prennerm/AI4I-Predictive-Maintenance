"""
Base Model Module for AI4I Predictive Maintenance Project

This module provides the abstract base class and interfaces for all machine learning models.
It defines the common contract that all model implementations must follow, ensuring
consistency across different model types (traditional ML, neural networks, ensembles).

Author: AI4I Team
Date: August 2025
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings

# Configure logging
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models in the AI4I project.
    
    This class defines the interface that all model implementations must follow,
    ensuring consistency and interoperability across different model types.
    All concrete model classes must inherit from this base class.
    """
    
    def __init__(self, model_name: str, model_params: Optional[Dict] = None, 
                 random_state: int = 42):
        """
        Initialize the base model.
        
        Args:
            model_name (str): Name identifier for the model
            model_params (Dict): Model-specific parameters
            random_state (int): Random state for reproducibility
        """
        self.model_name = model_name
        self.model_params = model_params or {}
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.target_name = None
        self.training_time = None
        self.model_metadata = {
            'created_at': datetime.now().isoformat(),
            'model_type': self.__class__.__name__,
            'model_name': model_name,
            'random_state': random_state,
            'sklearn_version': None,
            'feature_count': None,
            'training_samples': None
        }
        
        # Performance tracking
        self.training_metrics = {}
        self.validation_metrics = {}
        self.feature_importance = None
        
        logger.info(f"Initialized {self.__class__.__name__}: {model_name}")
    
    @abstractmethod
    def build_model(self) -> None:
        """
        Build/initialize the underlying model with specified parameters.
        This method must be implemented by all concrete model classes.
        """
        pass
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features (optional)
            y_val (pd.Series): Validation target (optional)
            
        Returns:
            Dict[str, Any]: Training metrics and results
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on the provided data.
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities (for classification models).
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        pass
    
    def validate_input_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Validate input data format and consistency.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable (optional)
            
        Raises:
            ValueError: If data validation fails
        """
        # Check if X is DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Features must be pandas DataFrame, got {type(X)}")
        
        # Check for empty dataset
        if X.empty:
            raise ValueError("Feature matrix cannot be empty")
        
        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            raise ValueError("Feature matrix contains infinite values")
        
        # Check target if provided
        if y is not None:
            if not isinstance(y, pd.Series):
                raise ValueError(f"Target must be pandas Series, got {type(y)}")
            
            if len(X) != len(y):
                raise ValueError(f"Feature and target length mismatch: {len(X)} vs {len(y)}")
        
        # Check feature consistency with trained model
        if self.feature_names is not None:
            if list(X.columns) != self.feature_names:
                missing_features = set(self.feature_names) - set(X.columns)
                extra_features = set(X.columns) - set(self.feature_names)
                
                error_msg = "Feature mismatch with trained model."
                if missing_features:
                    error_msg += f" Missing: {missing_features}"
                if extra_features:
                    error_msg += f" Extra: {extra_features}"
                
                raise ValueError(error_msg)
        
        logger.debug(f"Input validation passed for {X.shape[0]} samples, {X.shape[1]} features")
    
    def set_training_data_info(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Store information about training data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
        """
        self.feature_names = list(X_train.columns)
        self.target_name = y_train.name if y_train.name else 'target'
        
        # Update metadata
        self.model_metadata.update({
            'feature_count': len(self.feature_names),
            'training_samples': len(X_train),
            'feature_names': self.feature_names,
            'target_name': self.target_name
        })
        
        logger.info(f"Training data info set: {len(self.feature_names)} features, {len(X_train)} samples")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores if available.
        
        Returns:
            pd.DataFrame: Feature importance scores, or None if not available
        """
        if self.feature_importance is None:
            logger.warning(f"Feature importance not available for {self.model_name}")
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information and metadata.
        
        Returns:
            Dict[str, Any]: Model information dictionary
        """
        info = {
            'model_metadata': self.model_metadata.copy(),
            'model_params': self.model_params.copy(),
            'is_trained': self.is_trained,
            'training_time': self.training_time,
            'training_metrics': self.training_metrics.copy(),
            'validation_metrics': self.validation_metrics.copy()
        }
        
        if self.feature_names:
            info['feature_info'] = {
                'feature_count': len(self.feature_names),
                'feature_names': self.feature_names.copy()
            }
        
        return info
    
    def save_model(self, filepath: Union[str, Path], 
                   include_metadata: bool = True) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath (Union[str, Path]): Path to save the model
            include_metadata (bool): Whether to save metadata separately
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        try:
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
        
        # Save metadata if requested
        if include_metadata:
            metadata_path = filepath.with_suffix('.metadata.json')
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(self.get_model_info(), f, indent=2, default=str)
                logger.info(f"Model metadata saved to: {metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to save metadata: {str(e)}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path], 
                   load_metadata: bool = True) -> 'BaseModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath (Union[str, Path]): Path to the saved model
            load_metadata (bool): Whether to load metadata
            
        Returns:
            BaseModel: Loaded model instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # This method should be overridden by concrete classes
        # to properly reconstruct the model instance
        raise NotImplementedError("load_model must be implemented by concrete model classes")
    
    def reset_model(self) -> None:
        """
        Reset the model to untrained state.
        """
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.target_name = None
        self.training_time = None
        self.training_metrics = {}
        self.validation_metrics = {}
        self.feature_importance = None
        
        # Reset metadata timestamps
        self.model_metadata['created_at'] = datetime.now().isoformat()
        
        logger.info(f"Model {self.model_name} reset to untrained state")
    
    def __str__(self) -> str:
        """String representation of the model."""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}(name='{self.model_name}', status='{status}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return (f"{self.__class__.__name__}("
                f"model_name='{self.model_name}', "
                f"is_trained={self.is_trained}, "
                f"features={len(self.feature_names) if self.feature_names else 0})")


class ClassificationModel(BaseModel):
    """
    Abstract base class for classification models.
    
    Extends BaseModel with classification-specific functionality.
    """
    
    def __init__(self, model_name: str, model_params: Optional[Dict] = None,
                 random_state: int = 42):
        """
        Initialize classification model.
        
        Args:
            model_name (str): Name identifier for the model
            model_params (Dict): Model-specific parameters
            random_state (int): Random state for reproducibility
        """
        super().__init__(model_name, model_params, random_state)
        self.classes_ = None
        self.n_classes_ = None
    
    def validate_classification_data(self, y: pd.Series) -> None:
        """
        Validate target data for classification tasks.
        
        Args:
            y (pd.Series): Target variable
            
        Raises:
            ValueError: If validation fails
        """
        if y.isnull().any():
            raise ValueError("Target contains missing values")
        
        # Check if binary classification
        unique_values = y.nunique()
        if unique_values < 2:
            raise ValueError("Target must have at least 2 classes")
        
        # Store class information
        self.classes_ = sorted(y.unique())
        self.n_classes_ = len(self.classes_)
        
        logger.info(f"Classification target validated: {self.n_classes_} classes")
    
    def get_classification_report(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Generate comprehensive classification report.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): True labels
            
        Returns:
            Dict[str, Any]: Classification report
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating report")
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        report = {
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'predicted_probabilities': y_proba,
            'predictions': y_pred,
            'true_labels': y.values
        }
        
        return report


class RegressionModel(BaseModel):
    """
    Abstract base class for regression models.
    
    Extends BaseModel with regression-specific functionality.
    """
    
    def __init__(self, model_name: str, model_params: Optional[Dict] = None,
                 random_state: int = 42):
        """
        Initialize regression model.
        
        Args:
            model_name (str): Name identifier for the model
            model_params (Dict): Model-specific parameters
            random_state (int): Random state for reproducibility
        """
        super().__init__(model_name, model_params, random_state)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Regression models don't have predict_proba.
        Returns empty array to maintain interface compatibility.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Empty array
        """
        logger.warning("predict_proba not applicable for regression models")
        return np.array([])
    
    def validate_regression_data(self, y: pd.Series) -> None:
        """
        Validate target data for regression tasks.
        
        Args:
            y (pd.Series): Target variable
            
        Raises:
            ValueError: If validation fails
        """
        if y.isnull().any():
            raise ValueError("Target contains missing values")
        
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Regression target must be numeric")
        
        logger.info(f"Regression target validated: range [{y.min():.3f}, {y.max():.3f}]")
    
    def get_regression_report(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Generate comprehensive regression report.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): True values
            
        Returns:
            Dict[str, Any]: Regression report
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating report")
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_pred = self.predict(X)
        
        report = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2_score': r2_score(y, y_pred),
            'predictions': y_pred,
            'true_values': y.values,
            'residuals': (y.values - y_pred).tolist()
        }
        
        return report


# Factory function for creating model instances
def create_model(model_type: str, model_name: str, 
                model_params: Optional[Dict] = None,
                random_state: int = 42) -> BaseModel:
    """
    Factory function to create model instances.
    
    Args:
        model_type (str): Type of model to create
        model_name (str): Name for the model instance
        model_params (Dict): Model parameters
        random_state (int): Random state
        
    Returns:
        BaseModel: Model instance
        
    Raises:
        ValueError: If model_type is not supported
    """
    # This will be expanded as we implement concrete model classes
    supported_models = {
        'random_forest': 'RandomForestModel',
        'xgboost': 'XGBoostModel',
        'logistic_regression': 'LogisticRegressionModel',
        'svm': 'SVMModel',
        'neural_network': 'NeuralNetworkModel'
    }
    
    if model_type not in supported_models:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: {list(supported_models.keys())}")
    
    logger.info(f"Model factory will create {model_type} model when implemented")
    
    # For now, return None - this will be implemented when we create concrete models
    raise NotImplementedError(f"Concrete implementation for {model_type} not yet available")


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing base model classes...")
    
    # Test model metadata
    class TestModel(ClassificationModel):
        def build_model(self): 
            self.model = "dummy_model"
        def train(self, X_train, y_train, X_val=None, y_val=None): 
            return {}
        def predict(self, X): 
            return np.zeros(len(X))
        def predict_proba(self, X): 
            return np.zeros((len(X), 2))
    
    # Create test instance
    test_model = TestModel("test_classifier", {"param1": "value1"})
    
    print(f"Model created: {test_model}")
    print(f"Model info: {test_model.get_model_info()}")
    
    # Test with sample data
    sample_X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10]
    })
    sample_y = pd.Series([0, 1, 0, 1, 0], name='target')
    
    # Test validation
    try:
        test_model.validate_input_data(sample_X, sample_y)
        test_model.validate_classification_data(sample_y)
        print("Data validation passed!")
    except Exception as e:
        print(f"Validation error: {e}")
    
    logger.info("Base model testing completed!")
