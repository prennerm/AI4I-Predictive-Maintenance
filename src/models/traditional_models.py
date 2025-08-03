"""
Traditional Machine Learning Models Module

This module provides concrete implementations of traditional ML algorithms
for the AI4I predictive maintenance project:
- Random Forest (Classification & Regression)
- Support Vector Machine (Classification & Regression)
- XGBoost (Classification & Regression)
- Logistic Regression
- Decision Tree

All models inherit from BaseModel classes and are fully compatible with
the ModelTrainer and ModelEvaluator frameworks.

Author: AI4I Project Team
Created: August 2025
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd

# Scikit-learn models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

# LightGBM (optional)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Install with: pip install lightgbm")

# Our base classes
from src.models.base_model import BaseModel, ClassificationModel, RegressionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestModel(ClassificationModel):
    """
    Random Forest implementation with optimized hyperparameters for predictive maintenance.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 bootstrap: bool = True,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """
        Initialize Random Forest classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split node
            min_samples_leaf: Minimum samples required at leaf node
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap samples
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
            **kwargs: Additional parameters
        """
        super().__init__()
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
        
        self.hyperparameter_space = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5]
        }
        
        logger.info("RandomForestModel initialized")
    
    def get_hyperparameter_space(self) -> Dict[str, List[Any]]:
        """Get hyperparameter space for tuning."""
        return self.hyperparameter_space
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.feature_importances_


class RandomForestRegressorModel(RegressionModel):
    """Random Forest Regressor implementation."""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 bootstrap: bool = True,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """Initialize Random Forest regressor."""
        super().__init__()
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
        
        self.hyperparameter_space = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5]
        }


class SVMModel(ClassificationModel):
    """
    Support Vector Machine implementation optimized for predictive maintenance.
    """
    
    def __init__(self,
                 C: float = 1.0,
                 kernel: str = 'rbf',
                 degree: int = 3,
                 gamma: str = 'scale',
                 probability: bool = True,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize SVM classifier.
        
        Args:
            C: Regularization parameter
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            degree: Degree for polynomial kernel
            gamma: Kernel coefficient
            probability: Whether to enable probability estimates
            random_state: Random state for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__()
        
        self.model = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
            **kwargs
        )
        
        self.hyperparameter_space = {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4, 5]  # Only relevant for poly kernel
        }
        
        logger.info("SVMModel initialized")
    
    def get_hyperparameter_space(self) -> Dict[str, List[Any]]:
        """Get hyperparameter space for tuning."""
        return self.hyperparameter_space


class SVMRegressorModel(RegressionModel):
    """Support Vector Regressor implementation."""
    
    def __init__(self,
                 C: float = 1.0,
                 kernel: str = 'rbf',
                 degree: int = 3,
                 gamma: str = 'scale',
                 epsilon: float = 0.1,
                 **kwargs):
        """Initialize SVM regressor."""
        super().__init__()
        
        self.model = SVR(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            epsilon=epsilon,
            **kwargs
        )
        
        self.hyperparameter_space = {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2, 0.5, 1.0]
        }


class XGBoostModel(ClassificationModel):
    """
    XGBoost implementation optimized for predictive maintenance.
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 reg_alpha: float = 0,
                 reg_lambda: float = 1,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """
        Initialize XGBoost classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
            **kwargs: Additional parameters
        """
        super().__init__()
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric='logloss',  # Suppress warning
            **kwargs
        )
        
        self.hyperparameter_space = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [1, 1.5, 2, 5]
        }
        
        logger.info("XGBoostModel initialized")
    
    def get_hyperparameter_space(self) -> Dict[str, List[Any]]:
        """Get hyperparameter space for tuning."""
        return self.hyperparameter_space
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.feature_importances_


class XGBoostRegressorModel(RegressionModel):
    """XGBoost Regressor implementation."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 reg_alpha: float = 0,
                 reg_lambda: float = 1,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """Initialize XGBoost regressor."""
        super().__init__()
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")
        
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
        
        self.hyperparameter_space = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [1, 1.5, 2, 5]
        }


class LightGBMModel(ClassificationModel):
    """
    LightGBM implementation optimized for predictive maintenance.
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = -1,
                 learning_rate: float = 0.1,
                 num_leaves: int = 31,
                 feature_fraction: float = 0.9,
                 bagging_fraction: float = 0.8,
                 bagging_freq: int = 5,
                 min_child_samples: int = 20,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """Initialize LightGBM classifier."""
        super().__init__()
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Install with: pip install lightgbm")
        
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            min_child_samples=min_child_samples,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,  # Suppress warnings
            **kwargs
        )
        
        self.hyperparameter_space = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [-1, 5, 10, 15],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [15, 31, 50, 100],
            'feature_fraction': [0.8, 0.9, 1.0],
            'bagging_fraction': [0.8, 0.9, 1.0],
            'min_child_samples': [10, 20, 30]
        }
        
        logger.info("LightGBMModel initialized")


class LogisticRegressionModel(ClassificationModel):
    """
    Logistic Regression implementation with L1/L2 regularization.
    """
    
    def __init__(self,
                 penalty: str = 'l2',
                 C: float = 1.0,
                 solver: str = 'liblinear',
                 max_iter: int = 1000,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Logistic Regression classifier.
        
        Args:
            penalty: Regularization type ('l1', 'l2', 'elasticnet', None)
            C: Inverse of regularization strength
            solver: Algorithm to use ('liblinear', 'lbfgs', 'saga')
            max_iter: Maximum number of iterations
            random_state: Random state for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__()
        
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
        
        self.hyperparameter_space = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        logger.info("LogisticRegressionModel initialized")


class DecisionTreeModel(ClassificationModel):
    """
    Decision Tree implementation optimized for interpretability.
    """
    
    def __init__(self,
                 criterion: str = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[str] = None,
                 random_state: int = 42,
                 **kwargs):
        """Initialize Decision Tree classifier."""
        super().__init__()
        
        self.model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            **kwargs
        )
        
        self.hyperparameter_space = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': [None, 'sqrt', 'log2']
        }
        
        logger.info("DecisionTreeModel initialized")


class KNeighborsModel(ClassificationModel):
    """
    K-Nearest Neighbors implementation.
    """
    
    def __init__(self,
                 n_neighbors: int = 5,
                 weights: str = 'uniform',
                 algorithm: str = 'auto',
                 metric: str = 'minkowski',
                 p: int = 2,
                 n_jobs: int = -1,
                 **kwargs):
        """Initialize KNN classifier."""
        super().__init__()
        
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            metric=metric,
            p=p,
            n_jobs=n_jobs,
            **kwargs
        )
        
        self.hyperparameter_space = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'metric': ['minkowski', 'euclidean', 'manhattan'],
            'p': [1, 2]
        }
        
        logger.info("KNeighborsModel initialized")


class NaiveBayesModel(ClassificationModel):
    """
    Gaussian Naive Bayes implementation.
    """
    
    def __init__(self, var_smoothing: float = 1e-9, **kwargs):
        """Initialize Naive Bayes classifier."""
        super().__init__()
        
        self.model = GaussianNB(
            var_smoothing=var_smoothing,
            **kwargs
        )
        
        self.hyperparameter_space = {
            'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
        }
        
        logger.info("NaiveBayesModel initialized")


# Regression models
class LinearRegressionModel(RegressionModel):
    """Linear Regression implementation."""
    
    def __init__(self, fit_intercept: bool = True, **kwargs):
        """Initialize Linear Regression."""
        super().__init__()
        
        self.model = LinearRegression(
            fit_intercept=fit_intercept,
            **kwargs
        )
        
        self.hyperparameter_space = {
            'fit_intercept': [True, False]
        }


class RidgeRegressionModel(RegressionModel):
    """Ridge Regression with L2 regularization."""
    
    def __init__(self, alpha: float = 1.0, random_state: int = 42, **kwargs):
        """Initialize Ridge Regression."""
        super().__init__()
        
        self.model = Ridge(
            alpha=alpha,
            random_state=random_state,
            **kwargs
        )
        
        self.hyperparameter_space = {
            'alpha': [0.01, 0.1, 1, 10, 100, 1000]
        }


class LassoRegressionModel(RegressionModel):
    """Lasso Regression with L1 regularization."""
    
    def __init__(self, alpha: float = 1.0, random_state: int = 42, **kwargs):
        """Initialize Lasso Regression."""
        super().__init__()
        
        self.model = Lasso(
            alpha=alpha,
            random_state=random_state,
            **kwargs
        )
        
        self.hyperparameter_space = {
            'alpha': [0.01, 0.1, 1, 10, 100]
        }


# Model Factory Classes

class TraditionalModelFactory:
    """
    Factory class for creating traditional ML models.
    """
    
    # Classification models registry
    CLASSIFICATION_MODELS = {
        'random_forest': RandomForestModel,
        'svm': SVMModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'logistic_regression': LogisticRegressionModel,
        'decision_tree': DecisionTreeModel,
        'knn': KNeighborsModel,
        'naive_bayes': NaiveBayesModel
    }
    
    # Regression models registry
    REGRESSION_MODELS = {
        'random_forest': RandomForestRegressorModel,
        'svm': SVMRegressorModel,
        'xgboost': XGBoostRegressorModel,
        'linear_regression': LinearRegressionModel,
        'ridge': RidgeRegressionModel,
        'lasso': LassoRegressionModel,
        'knn': KNeighborsRegressor  # Note: Need to implement wrapper
    }
    
    @classmethod
    def create_classifier(cls, model_type: str, **kwargs) -> ClassificationModel:
        """
        Create a classification model.
        
        Args:
            model_type: Type of model to create
            **kwargs: Model parameters
            
        Returns:
            Initialized classification model
        """
        if model_type not in cls.CLASSIFICATION_MODELS:
            available_models = list(cls.CLASSIFICATION_MODELS.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
        
        model_class = cls.CLASSIFICATION_MODELS[model_type]
        
        # Handle optional dependencies
        if model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")
        if model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Install with: pip install lightgbm")
        
        return model_class(**kwargs)
    
    @classmethod
    def create_regressor(cls, model_type: str, **kwargs) -> RegressionModel:
        """
        Create a regression model.
        
        Args:
            model_type: Type of model to create
            **kwargs: Model parameters
            
        Returns:
            Initialized regression model
        """
        if model_type not in cls.REGRESSION_MODELS:
            available_models = list(cls.REGRESSION_MODELS.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
        
        model_class = cls.REGRESSION_MODELS[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> Dict[str, List[str]]:
        """Get list of available models."""
        return {
            'classification': list(cls.CLASSIFICATION_MODELS.keys()),
            'regression': list(cls.REGRESSION_MODELS.keys())
        }
    
    @classmethod
    def create_model_suite(cls, problem_type: str = 'classification') -> List[BaseModel]:
        """
        Create a suite of models for comparison.
        
        Args:
            problem_type: 'classification' or 'regression'
            
        Returns:
            List of initialized models
        """
        models = []
        
        if problem_type == 'classification':
            # Core models that should always work
            core_models = ['random_forest', 'svm', 'logistic_regression', 'decision_tree']
            
            for model_type in core_models:
                try:
                    model = cls.create_classifier(model_type)
                    models.append(model)
                except Exception as e:
                    logger.warning(f"Could not create {model_type}: {str(e)}")
            
            # Optional models (if dependencies available)
            optional_models = ['xgboost', 'lightgbm']
            for model_type in optional_models:
                try:
                    model = cls.create_classifier(model_type)
                    models.append(model)
                    logger.info(f"Added optional model: {model_type}")
                except ImportError:
                    logger.info(f"Skipping {model_type} (dependency not available)")
                except Exception as e:
                    logger.warning(f"Could not create {model_type}: {str(e)}")
        
        elif problem_type == 'regression':
            core_models = ['random_forest', 'svm', 'linear_regression', 'ridge', 'lasso']
            
            for model_type in core_models:
                try:
                    model = cls.create_regressor(model_type)
                    models.append(model)
                except Exception as e:
                    logger.warning(f"Could not create {model_type}: {str(e)}")
            
            # Optional XGBoost regressor
            try:
                model = cls.create_regressor('xgboost')
                models.append(model)
            except ImportError:
                logger.info("Skipping XGBoost regressor (dependency not available)")
        
        logger.info(f"Created model suite with {len(models)} models for {problem_type}")
        return models


# Convenience functions for quick model creation

def create_baseline_models(problem_type: str = 'classification') -> List[BaseModel]:
    """
    Create a set of baseline models for quick comparison.
    
    Args:
        problem_type: 'classification' or 'regression'
        
    Returns:
        List of baseline models
    """
    factory = TraditionalModelFactory()
    
    if problem_type == 'classification':
        models = [
            factory.create_classifier('random_forest', n_estimators=50),
            factory.create_classifier('svm', C=1.0, kernel='rbf'),
            factory.create_classifier('logistic_regression', C=1.0)
        ]
    else:
        models = [
            factory.create_regressor('random_forest', n_estimators=50),
            factory.create_regressor('svm', C=1.0, kernel='rbf'),
            factory.create_regressor('linear_regression')
        ]
    
    return models


def create_optimized_models(problem_type: str = 'classification') -> List[BaseModel]:
    """
    Create models with optimized hyperparameters for predictive maintenance.
    
    Args:
        problem_type: 'classification' or 'regression'
        
    Returns:
        List of optimized models
    """
    factory = TraditionalModelFactory()
    
    if problem_type == 'classification':
        models = [
            # Optimized Random Forest
            factory.create_classifier(
                'random_forest',
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt'
            ),
            
            # Optimized SVM
            factory.create_classifier(
                'svm',
                C=10,
                kernel='rbf',
                gamma='scale',
                probability=True
            ),
            
            # Optimized Logistic Regression
            factory.create_classifier(
                'logistic_regression',
                C=1.0,
                penalty='l2',
                solver='liblinear'
            )
        ]
        
        # Add XGBoost if available
        try:
            xgb_model = factory.create_classifier(
                'xgboost',
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9
            )
            models.append(xgb_model)
        except ImportError:
            logger.info("XGBoost not available for optimized models")
    
    else:
        models = [
            factory.create_regressor(
                'random_forest',
                n_estimators=200,
                max_depth=15,
                min_samples_split=5
            ),
            factory.create_regressor('ridge', alpha=1.0),
            factory.create_regressor('lasso', alpha=0.1)
        ]
    
    return models


# Hyperparameter spaces for different model types
def get_hyperparameter_spaces(problem_type: str = 'classification') -> Dict[str, Dict[str, List[Any]]]:
    """
    Get comprehensive hyperparameter spaces for model tuning.
    
    Args:
        problem_type: 'classification' or 'regression'
        
    Returns:
        Dictionary mapping model names to hyperparameter spaces
    """
    factory = TraditionalModelFactory()
    
    spaces = {}
    
    if problem_type == 'classification':
        model_types = ['random_forest', 'svm', 'logistic_regression', 'decision_tree']
    else:
        model_types = ['random_forest', 'svm', 'linear_regression', 'ridge', 'lasso']
    
    for model_type in model_types:
        try:
            if problem_type == 'classification':
                model = factory.create_classifier(model_type)
            else:
                model = factory.create_regressor(model_type)
            
            spaces[model_type] = model.get_hyperparameter_space()
        except Exception as e:
            logger.warning(f"Could not get hyperparameter space for {model_type}: {str(e)}")
    
    return spaces


# Testing and validation functions
def test_traditional_models():
    """Test function for traditional models."""
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    
    print("Testing Traditional Models...")
    
    # Test classification models
    X_cls, y_cls = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    
    factory = TraditionalModelFactory()
    
    # Test model creation
    rf_model = factory.create_classifier('random_forest')
    assert isinstance(rf_model, RandomForestModel)
    
    # Test training and prediction
    rf_model.train(X_train_cls, y_train_cls)
    predictions = rf_model.predict(X_test_cls)
    probabilities = rf_model.predict_proba(X_test_cls)
    
    assert len(predictions) == len(X_test_cls)
    assert probabilities.shape == (len(X_test_cls), 2)
    
    # Test model suite creation
    model_suite = factory.create_model_suite('classification')
    assert len(model_suite) >= 3  # At least core models
    
    # Test regression models
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    ridge_model = factory.create_regressor('ridge')
    ridge_model.train(X_train_reg, y_train_reg)
    reg_predictions = ridge_model.predict(X_test_reg)
    
    assert len(reg_predictions) == len(X_test_reg)
    
    print("All tests passed!")


if __name__ == "__main__":
    test_traditional_models()
