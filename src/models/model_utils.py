"""
Model Utilities Module

This module provides comprehensive utility functions for model operations including:
- Model persistence (save/load with metadata)
- Performance evaluation and metrics calculation
- Model comparison and selection
- Hyperparameter optimization utilities
- Model validation and cross-validation
- Feature importance analysis
- Model interpretation utilities

Author: AI4I Project Team
Created: August 2025
"""

import os
import json
import pickle
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, mean_squared_error,
    mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPersistence:
    """Handles model saving and loading with comprehensive metadata."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize ModelPersistence.
        
        Args:
            models_dir: Directory to store model artifacts
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def save_model(self, 
                   model: Any, 
                   model_name: str,
                   metadata: Dict[str, Any] = None,
                   feature_names: List[str] = None,
                   model_version: str = "1.0") -> str:
        """
        Save model with comprehensive metadata.
        
        Args:
            model: Trained model object
            model_name: Name for the model
            metadata: Additional metadata to store
            feature_names: List of feature names used for training
            model_version: Version of the model
            
        Returns:
            Path to saved model directory
        """
        try:
            # Create model directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = self.models_dir / f"{model_name}_{timestamp}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model using both pickle and joblib for compatibility
            model_path = model_dir / "model.pkl"
            joblib_path = model_dir / "model.joblib"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            joblib.dump(model, joblib_path)
            
            # Prepare comprehensive metadata
            model_metadata = {
                "model_name": model_name,
                "model_version": model_version,
                "timestamp": timestamp,
                "model_type": type(model).__name__,
                "model_module": type(model).__module__,
                "feature_names": feature_names or [],
                "feature_count": len(feature_names) if feature_names else 0,
                "saved_by": "AI4I_ModelUtils",
                "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
                "model_size_bytes": os.path.getsize(model_path) if model_path.exists() else 0
            }
            
            # Add custom metadata
            if metadata:
                model_metadata.update(metadata)
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2, default=str)
            
            # Save feature names separately for easy access
            if feature_names:
                features_path = model_dir / "features.json"
                with open(features_path, 'w') as f:
                    json.dump(feature_names, f, indent=2)
            
            logger.info(f"Model saved successfully to: {model_dir}")
            return str(model_dir)
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
            raise
    
    def load_model(self, model_path: str, use_joblib: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Load model with metadata.
        
        Args:
            model_path: Path to model directory or file
            use_joblib: Whether to use joblib for loading
            
        Returns:
            Tuple of (model, metadata)
        """
        try:
            model_path = Path(model_path)
            
            # Determine model file path
            if model_path.is_dir():
                model_file = model_path / ("model.joblib" if use_joblib else "model.pkl")
                metadata_file = model_path / "metadata.json"
            else:
                model_file = model_path
                metadata_file = model_path.parent / "metadata.json"
            
            # Load model
            if use_joblib and model_file.with_suffix('.joblib').exists():
                model = joblib.load(model_file.with_suffix('.joblib'))
            else:
                with open(model_file.with_suffix('.pkl'), 'rb') as f:
                    model = pickle.load(f)
            
            # Load metadata
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            logger.info(f"Model loaded successfully from: {model_path}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise
    
    def list_saved_models(self) -> List[Dict[str, Any]]:
        """
        List all saved models with their metadata.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        metadata['model_path'] = str(model_dir)
                        models.append(metadata)
                    except Exception as e:
                        logger.warning(f"Could not read metadata for {model_dir}: {str(e)}")
        
        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return models


class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray, 
                              y_pred: np.ndarray, 
                              y_proba: np.ndarray = None,
                              class_names: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive classification evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            class_names: Names of classes
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            metrics = {}
            
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
            
            # Per-class metrics
            metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None).tolist()
            metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None).tolist()
            metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None).tolist()
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # ROC AUC (if probabilities available)
            if y_proba is not None:
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                elif len(np.unique(y_true)) == 2:
                    # Binary with single probability
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                else:
                    # Multi-class
                    try:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                    except ValueError:
                        logger.warning("Could not compute ROC AUC for multi-class problem")
            
            # Classification report
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            metrics['classification_report'] = report
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in classification evaluation: {str(e)}")
            raise
    
    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive regression evaluation.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            metrics = {}
            
            # Basic metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Additional metrics
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['residual_std'] = np.std(y_true - y_pred)
            
            # Percentile-based metrics
            residuals = np.abs(y_true - y_pred)
            metrics['median_abs_error'] = np.median(residuals)
            metrics['q95_abs_error'] = np.percentile(residuals, 95)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in regression evaluation: {str(e)}")
            raise
    
    @staticmethod
    def cross_validate_model(model: Any,
                           X: np.ndarray,
                           y: np.ndarray,
                           cv: int = 5,
                           scoring: str = 'accuracy',
                           stratify: bool = True) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target
            cv: Number of folds
            scoring: Scoring metric
            stratify: Whether to use stratified CV
            
        Returns:
            Cross-validation results
        """
        try:
            # Choose cross-validation strategy
            if stratify and len(np.unique(y)) > 1:
                cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            else:
                cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
            
            # Perform cross-validation
            scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring)
            
            results = {
                'scores': scores.tolist(),
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'min_score': scores.min(),
                'max_score': scores.max(),
                'cv_folds': cv,
                'scoring_metric': scoring
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise


class ModelComparison:
    """Utilities for comparing multiple models."""
    
    def __init__(self):
        self.results = []
    
    def add_model_result(self, 
                        model_name: str,
                        metrics: Dict[str, Any],
                        training_time: float = None,
                        prediction_time: float = None):
        """
        Add model evaluation results.
        
        Args:
            model_name: Name of the model
            metrics: Evaluation metrics
            training_time: Time taken to train (seconds)
            prediction_time: Time taken to predict (seconds)
        """
        result = {
            'model_name': model_name,
            'metrics': metrics,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(result)
    
    def get_comparison_table(self, 
                           primary_metric: str = 'accuracy',
                           include_times: bool = True) -> pd.DataFrame:
        """
        Create comparison table of all models.
        
        Args:
            primary_metric: Main metric to sort by
            include_times: Whether to include timing information
            
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for result in self.results:
            row = {'Model': result['model_name']}
            
            # Add metrics
            metrics = result['metrics']
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        row[key] = value
            
            # Add timing information
            if include_times:
                row['Training_Time'] = result.get('training_time', 'N/A')
                row['Prediction_Time'] = result.get('prediction_time', 'N/A')
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric if available
        if primary_metric in df.columns:
            df = df.sort_values(primary_metric, ascending=False)
        
        return df
    
    def get_best_model(self, metric: str = 'accuracy', higher_is_better: bool = True) -> Dict[str, Any]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to optimize
            higher_is_better: Whether higher values are better
            
        Returns:
            Best model result
        """
        if not self.results:
            return None
        
        best_result = None
        best_score = float('-inf') if higher_is_better else float('inf')
        
        for result in self.results:
            metrics = result.get('metrics', {})
            if metric in metrics:
                score = metrics[metric]
                if isinstance(score, (int, float)):
                    if (higher_is_better and score > best_score) or \
                       (not higher_is_better and score < best_score):
                        best_score = score
                        best_result = result
        
        return best_result


class FeatureImportanceAnalyzer:
    """Analyze and visualize feature importance across different models."""
    
    @staticmethod
    def get_feature_importance(model: Any, 
                             feature_names: List[str] = None) -> Dict[str, float]:
        """
        Extract feature importance from a trained model.
        
        Args:
            model: Trained model
            feature_names: Names of features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            importance_scores = None
            
            # Try different methods to get feature importance
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                coef = model.coef_
                if coef.ndim > 1:
                    importance_scores = np.mean(np.abs(coef), axis=0)
                else:
                    importance_scores = np.abs(coef)
            elif hasattr(model, 'named_steps'):
                # For pipelines, try to get from the last step
                last_step = list(model.named_steps.values())[-1]
                return FeatureImportanceAnalyzer.get_feature_importance(last_step, feature_names)
            
            if importance_scores is not None:
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(len(importance_scores))]
                
                # Create importance dictionary
                importance_dict = dict(zip(feature_names, importance_scores))
                
                # Sort by importance
                importance_dict = dict(sorted(importance_dict.items(), 
                                            key=lambda x: x[1], reverse=True))
                
                return importance_dict
            else:
                logger.warning(f"Could not extract feature importance from {type(model).__name__}")
                return {}
                
        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            return {}
    
    @staticmethod
    def plot_feature_importance(importance_dict: Dict[str, float],
                              top_n: int = 20,
                              figsize: Tuple[int, int] = (10, 8),
                              title: str = "Feature Importance") -> plt.Figure:
        """
        Create feature importance plot.
        
        Args:
            importance_dict: Feature importance dictionary
            top_n: Number of top features to show
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if not importance_dict:
            logger.warning("No feature importance data to plot")
            return None
        
        # Get top N features
        top_features = dict(list(importance_dict.items())[:top_n])
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(features)), importances)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        
        # Invert y-axis to show most important at top
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig


class ModelDiagnostics:
    """Advanced model diagnostics and validation utilities."""
    
    @staticmethod
    def learning_curve_analysis(model: Any,
                              X: np.ndarray,
                              y: np.ndarray,
                              train_sizes: np.ndarray = None,
                              cv: int = 5,
                              scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Analyze model learning curves.
        
        Args:
            model: Model to analyze
            X: Features
            y: Target
            train_sizes: Training set sizes to evaluate
            cv: Cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Learning curve data
        """
        from sklearn.model_selection import learning_curve
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring,
                n_jobs=-1, random_state=42
            )
            
            results = {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                'train_scores_std': np.std(train_scores, axis=1).tolist(),
                'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
                'val_scores_std': np.std(val_scores, axis=1).tolist(),
                'scoring_metric': scoring
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in learning curve analysis: {str(e)}")
            raise
    
    @staticmethod
    def validation_curve_analysis(model: Any,
                                X: np.ndarray,
                                y: np.ndarray,
                                param_name: str,
                                param_range: List[Any],
                                cv: int = 5,
                                scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Analyze validation curves for hyperparameter tuning.
        
        Args:
            model: Model to analyze
            X: Features
            y: Target
            param_name: Parameter name to vary
            param_range: Range of parameter values
            cv: Cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Validation curve data
        """
        from sklearn.model_selection import validation_curve
        
        try:
            train_scores, val_scores = validation_curve(
                model, X, y, param_name=param_name, param_range=param_range,
                cv=cv, scoring=scoring, n_jobs=-1
            )
            
            results = {
                'param_name': param_name,
                'param_range': [str(p) for p in param_range],
                'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                'train_scores_std': np.std(train_scores, axis=1).tolist(),
                'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
                'val_scores_std': np.std(val_scores, axis=1).tolist(),
                'scoring_metric': scoring
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in validation curve analysis: {str(e)}")
            raise


def safe_model_operation(operation_name: str):
    """
    Decorator for safe model operations with error handling.
    
    Args:
        operation_name: Name of the operation for logging
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                logger.info(f"Starting {operation_name}...")
                result = func(*args, **kwargs)
                logger.info(f"{operation_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Error in {operation_name}: {str(e)}")
                raise
        return wrapper
    return decorator


# Utility functions for common operations

def ensure_reproducibility(seed: int = 42):
    """Ensure reproducible results across all libraries."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def format_model_summary(model_info: Dict[str, Any]) -> str:
    """
    Format model information for display.
    
    Args:
        model_info: Model information dictionary
        
    Returns:
        Formatted string representation
    """
    summary = []
    summary.append(f"Model: {model_info.get('model_name', 'Unknown')}")
    summary.append(f"Type: {model_info.get('model_type', 'Unknown')}")
    summary.append(f"Version: {model_info.get('model_version', 'Unknown')}")
    summary.append(f"Features: {model_info.get('feature_count', 'Unknown')}")
    summary.append(f"Timestamp: {model_info.get('timestamp', 'Unknown')}")
    
    if 'metrics' in model_info:
        summary.append("\nKey Metrics:")
        metrics = model_info['metrics']
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                summary.append(f"  {key}: {value:.4f}")
    
    return "\n".join(summary)


def validate_model_inputs(X: np.ndarray, y: np.ndarray = None) -> bool:
    """
    Validate model inputs for common issues.
    
    Args:
        X: Feature matrix
        y: Target vector (optional)
        
    Returns:
        True if inputs are valid
        
    Raises:
        ValueError: If inputs are invalid
    """
    if X is None or len(X) == 0:
        raise ValueError("Feature matrix X is empty or None")
    
    if np.any(np.isnan(X)):
        raise ValueError("Feature matrix X contains NaN values")
    
    if np.any(np.isinf(X)):
        raise ValueError("Feature matrix X contains infinite values")
    
    if y is not None:
        if len(y) != len(X):
            raise ValueError("X and y must have the same number of samples")
        
        if np.any(np.isnan(y)):
            raise ValueError("Target vector y contains NaN values")
        
        if np.any(np.isinf(y)):
            raise ValueError("Target vector y contains infinite values")
    
    return True


# Example usage and testing functions
def test_model_utils():
    """Test function for model utilities."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    print("Testing Model Utils...")
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Test evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification(y_test, y_pred, y_proba)
    print(f"Classification metrics calculated: {len(metrics)} metrics")
    
    # Test feature importance
    analyzer = FeatureImportanceAnalyzer()
    importance = analyzer.get_feature_importance(model)
    print(f"Feature importance extracted: {len(importance)} features")
    
    # Test model persistence
    persistence = ModelPersistence()
    model_path = persistence.save_model(model, "test_model", metadata={"test": True})
    loaded_model, metadata = persistence.load_model(model_path)
    print(f"Model saved and loaded successfully")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_model_utils()
