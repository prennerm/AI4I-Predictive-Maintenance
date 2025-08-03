"""
Model Training Pipeline Module

This module provides a comprehensive training framework that orchestrates:
- Data preprocessing and feature engineering
- Model training with hyperparameter optimization
- Cross-validation and performance evaluation
- Model persistence and experiment tracking
- Integration with all existing components

Author: AI4I Project Team
Created: August 2025
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer

# Import our custom modules
from src.models.base_model import BaseModel, ClassificationModel, RegressionModel
from src.models.model_utils import (
    ModelPersistence, ModelEvaluator, ModelComparison,
    FeatureImportanceAnalyzer, ensure_reproducibility,
    validate_model_inputs, safe_model_operation
)
from src.features.feature_engineering import FeatureEngineer
from src.features.feature_selection import FeatureSelector
from src.utils.data_preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    
    # Data parameters
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    
    # Feature engineering
    feature_set: str = 'extended'  # 'baseline', 'extended', 'temperature_focus', etc.
    feature_selection: bool = True
    feature_selection_method: str = 'importance_hybrid'
    max_features: int = 50
    
    # Training parameters
    cross_validation: bool = True
    cv_folds: int = 5
    scoring_metric: str = 'f1_weighted'
    
    # Hyperparameter optimization
    hyperparameter_tuning: bool = True
    tuning_method: str = 'random_search'  # 'grid_search', 'random_search'
    tuning_iterations: int = 50
    tuning_cv_folds: int = 3
    
    # Model persistence
    save_models: bool = True
    save_best_only: bool = True
    model_version: str = "1.0"
    
    # Experiment tracking
    experiment_name: str = None
    track_experiments: bool = True
    
    # Performance
    n_jobs: int = -1
    verbose: int = 1


@dataclass
class TrainingResult:
    """Data class to store training results."""
    
    model_name: str
    model: Any
    train_score: float
    val_score: float
    test_score: float
    cv_scores: List[float]
    training_time: float
    hyperparams: Dict[str, Any]
    feature_importance: Dict[str, float]
    model_path: str = None
    metadata: Dict[str, Any] = None


class ModelTrainer:
    """
    Comprehensive model training pipeline that orchestrates all components.
    """
    
    def __init__(self, config: TrainingConfig = None):
        """
        Initialize ModelTrainer with configuration.
        
        Args:
            config: Training configuration object
        """
        self.config = config or TrainingConfig()
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector()
        self.model_persistence = ModelPersistence()
        self.model_comparison = ModelComparison()
        self.feature_analyzer = FeatureImportanceAnalyzer()
        
        # Storage for results
        self.training_results: List[TrainingResult] = []
        self.experiment_id = self._generate_experiment_id()
        
        # Ensure reproducibility
        ensure_reproducibility(self.config.random_state)
        
        logger.info(f"ModelTrainer initialized with experiment ID: {self.experiment_id}")
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config.experiment_name or "training"
        return f"{exp_name}_{timestamp}"
    
    @safe_model_operation("data_preparation")
    def prepare_data(self, 
                    data_path: str = None,
                    X: np.ndarray = None,
                    y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                   np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training including preprocessing and feature engineering.
        
        Args:
            data_path: Path to raw data file
            X: Feature matrix (alternative to data_path)
            y: Target vector (alternative to data_path)
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info("Starting data preparation...")
        
        # Load and preprocess data if path provided
        if data_path:
            logger.info(f"Loading data from: {data_path}")
            processed_data = self.preprocessor.preprocess_ai4i_data(
                data_path=data_path,
                output_dir="data/processed",
                return_processed_data=True
            )
            X = processed_data['features']
            y = processed_data['target']
        
        # Validate inputs
        validate_model_inputs(X, y)
        
        # Feature engineering
        logger.info(f"Applying feature engineering: {self.config.feature_set}")
        X_engineered = self.feature_engineer.get_feature_set(
            pd.DataFrame(X) if isinstance(X, np.ndarray) else X,
            feature_set=self.config.feature_set
        )
        
        # Convert to numpy if DataFrame
        if isinstance(X_engineered, pd.DataFrame):
            feature_names = X_engineered.columns.tolist()
            X_engineered = X_engineered.values
        else:
            feature_names = [f'feature_{i}' for i in range(X_engineered.shape[1])]
        
        # Feature selection
        if self.config.feature_selection:
            logger.info(f"Applying feature selection: {self.config.feature_selection_method}")
            X_selected, selected_features = self.feature_selector.select_best_features(
                X_engineered, y,
                method=self.config.feature_selection_method,
                k=min(self.config.max_features, X_engineered.shape[1])
            )
            
            # Update feature names
            if isinstance(selected_features, list) and all(isinstance(f, str) for f in selected_features):
                self.selected_feature_names = selected_features
            else:
                self.selected_feature_names = [feature_names[i] for i in selected_features]
            
            X_final = X_selected
        else:
            X_final = X_engineered
            self.selected_feature_names = feature_names
        
        # Data splitting
        logger.info("Splitting data into train/validation/test sets...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_final, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if self.config.stratify else None
        )
        
        # Second split: separate train and validation
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp if self.config.stratify else None
        )
        
        # Store data info
        self.data_info = {
            'total_samples': len(X_final),
            'total_features': X_final.shape[1],
            'selected_features': len(self.selected_feature_names),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'feature_names': self.selected_feature_names,
            'target_classes': len(np.unique(y)) if len(np.unique(y)) < 10 else 'continuous'
        }
        
        logger.info(f"Data preparation completed:")
        logger.info(f"  - Total samples: {self.data_info['total_samples']}")
        logger.info(f"  - Features: {self.data_info['selected_features']}")
        logger.info(f"  - Train: {self.data_info['train_samples']}, Val: {self.data_info['val_samples']}, Test: {self.data_info['test_samples']}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @safe_model_operation("model_training")
    def train_model(self,
                   model: BaseModel,
                   X_train: np.ndarray,
                   X_val: np.ndarray,
                   X_test: np.ndarray,
                   y_train: np.ndarray,
                   y_val: np.ndarray,
                   y_test: np.ndarray,
                   hyperparameter_space: Dict[str, Any] = None) -> TrainingResult:
        """
        Train a single model with comprehensive evaluation.
        
        Args:
            model: Model instance inheriting from BaseModel
            X_train, X_val, X_test: Feature matrices
            y_train, y_val, y_test: Target vectors
            hyperparameter_space: Parameter space for tuning
            
        Returns:
            TrainingResult object with all metrics and artifacts
        """
        model_name = model.__class__.__name__
        logger.info(f"Training model: {model_name}")
        
        start_time = time.time()
        
        # Hyperparameter optimization
        if self.config.hyperparameter_tuning and hyperparameter_space:
            logger.info(f"Starting hyperparameter tuning using {self.config.tuning_method}")
            
            if self.config.tuning_method == 'grid_search':
                search = GridSearchCV(
                    model.model,
                    hyperparameter_space,
                    cv=self.config.tuning_cv_folds,
                    scoring=self.config.scoring_metric,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbose
                )
            else:  # random_search
                search = RandomizedSearchCV(
                    model.model,
                    hyperparameter_space,
                    n_iter=self.config.tuning_iterations,
                    cv=self.config.tuning_cv_folds,
                    scoring=self.config.scoring_metric,
                    n_jobs=self.config.n_jobs,
                    random_state=self.config.random_state,
                    verbose=self.config.verbose
                )
            
            # Fit hyperparameter search
            search.fit(X_train, y_train)
            
            # Update model with best parameters
            model.model.set_params(**search.best_params_)
            best_hyperparams = search.best_params_
            
            logger.info(f"Best hyperparameters: {best_hyperparams}")
        else:
            best_hyperparams = model.model.get_params()
        
        # Train final model
        logger.info("Training final model...")
        model.train(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Cross-validation evaluation
        cv_scores = []
        if self.config.cross_validation:
            logger.info("Performing cross-validation...")
            cv_scores = cross_val_score(
                model.model, X_train, y_train,
                cv=self.config.cv_folds,
                scoring=self.config.scoring_metric,
                n_jobs=self.config.n_jobs
            ).tolist()
        
        # Predictions and evaluation
        logger.info("Evaluating model performance...")
        
        # Training score
        y_train_pred = model.predict(X_train)
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)
        else:
            y_train_proba = None
        
        # Validation score
        y_val_pred = model.predict(X_val)
        if hasattr(model, 'predict_proba'):
            y_val_proba = model.predict_proba(X_val)
        else:
            y_val_proba = None
        
        # Test score
        y_test_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test)
        else:
            y_test_proba = None
        
        # Calculate metrics using ModelEvaluator
        evaluator = ModelEvaluator()
        
        if isinstance(model, ClassificationModel):
            train_metrics = evaluator.evaluate_classification(y_train, y_train_pred, y_train_proba)
            val_metrics = evaluator.evaluate_classification(y_val, y_val_pred, y_val_proba)
            test_metrics = evaluator.evaluate_classification(y_test, y_test_pred, y_test_proba)
            
            # Use F1 score as primary metric
            train_score = train_metrics['f1_score']
            val_score = val_metrics['f1_score']
            test_score = test_metrics['f1_score']
        else:
            train_metrics = evaluator.evaluate_regression(y_train, y_train_pred)
            val_metrics = evaluator.evaluate_regression(y_val, y_val_pred)
            test_metrics = evaluator.evaluate_regression(y_test, y_test_pred)
            
            # Use R² as primary metric
            train_score = train_metrics['r2']
            val_score = val_metrics['r2']
            test_score = test_metrics['r2']
        
        # Feature importance analysis
        feature_importance = self.feature_analyzer.get_feature_importance(
            model.model, self.selected_feature_names
        )
        
        # Model persistence
        model_path = None
        if self.config.save_models:
            logger.info("Saving trained model...")
            
            metadata = {
                'experiment_id': self.experiment_id,
                'model_name': model_name,
                'train_score': train_score,
                'val_score': val_score,
                'test_score': test_score,
                'cv_scores': cv_scores,
                'training_time': training_time,
                'hyperparameters': best_hyperparams,
                'data_info': self.data_info,
                'config': asdict(self.config),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics
            }
            
            model_path = self.model_persistence.save_model(
                model.model,
                f"{model_name}_{self.experiment_id}",
                metadata=metadata,
                feature_names=self.selected_feature_names,
                model_version=self.config.model_version
            )
        
        # Create training result
        result = TrainingResult(
            model_name=model_name,
            model=model,
            train_score=train_score,
            val_score=val_score,
            test_score=test_score,
            cv_scores=cv_scores,
            training_time=training_time,
            hyperparams=best_hyperparams,
            feature_importance=feature_importance,
            model_path=model_path,
            metadata={
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'data_info': self.data_info
            }
        )
        
        # Add to comparison
        self.model_comparison.add_model_result(
            model_name=model_name,
            metrics={
                'train_score': train_score,
                'val_score': val_score,
                'test_score': test_score,
                'cv_mean': np.mean(cv_scores) if cv_scores else None,
                'cv_std': np.std(cv_scores) if cv_scores else None
            },
            training_time=training_time
        )
        
        # Store result
        self.training_results.append(result)
        
        logger.info(f"Model {model_name} training completed:")
        logger.info(f"  - Train score: {train_score:.4f}")
        logger.info(f"  - Val score: {val_score:.4f}")
        logger.info(f"  - Test score: {test_score:.4f}")
        logger.info(f"  - Training time: {training_time:.2f}s")
        
        return result
    
    def train_multiple_models(self,
                            models: List[Tuple[BaseModel, Dict[str, Any]]],
                            X_train: np.ndarray,
                            X_val: np.ndarray,
                            X_test: np.ndarray,
                            y_train: np.ndarray,
                            y_val: np.ndarray,
                            y_test: np.ndarray) -> List[TrainingResult]:
        """
        Train multiple models and compare their performance.
        
        Args:
            models: List of (model, hyperparameter_space) tuples
            X_train, X_val, X_test: Feature matrices
            y_train, y_val, y_test: Target vectors
            
        Returns:
            List of TrainingResult objects
        """
        logger.info(f"Training {len(models)} models...")
        
        results = []
        for i, (model, hyperparams) in enumerate(models, 1):
            logger.info(f"Training model {i}/{len(models)}: {model.__class__.__name__}")
            
            try:
                result = self.train_model(
                    model, X_train, X_val, X_test, y_train, y_val, y_test, hyperparams
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error training {model.__class__.__name__}: {str(e)}")
                continue
        
        logger.info(f"Training completed for {len(results)}/{len(models)} models")
        return results
    
    def get_training_summary(self) -> pd.DataFrame:
        """
        Get summary of all training results.
        
        Returns:
            DataFrame with model comparison
        """
        return self.model_comparison.get_comparison_table(
            primary_metric='val_score',
            include_times=True
        )
    
    def get_best_model(self, metric: str = 'val_score') -> TrainingResult:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to optimize ('val_score', 'test_score', etc.)
            
        Returns:
            Best TrainingResult
        """
        if not self.training_results:
            return None
        
        best_result = max(self.training_results, key=lambda x: getattr(x, metric))
        return best_result
    
    def save_experiment_results(self, output_dir: str = "reports/experiments") -> str:
        """
        Save comprehensive experiment results.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Path to saved results
        """
        output_path = Path(output_dir) / f"experiment_{self.experiment_id}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training summary
        summary_df = self.get_training_summary()
        summary_df.to_csv(output_path / "training_summary.csv", index=False)
        
        # Save detailed results
        detailed_results = []
        for result in self.training_results:
            result_dict = {
                'model_name': result.model_name,
                'train_score': result.train_score,
                'val_score': result.val_score,
                'test_score': result.test_score,
                'cv_scores': result.cv_scores,
                'training_time': result.training_time,
                'hyperparams': result.hyperparams,
                'model_path': result.model_path
            }
            if result.metadata:
                result_dict.update(result.metadata)
            detailed_results.append(result_dict)
        
        with open(output_path / "detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Save experiment configuration
        with open(output_path / "experiment_config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # Save data information
        with open(output_path / "data_info.json", 'w') as f:
            json.dump(self.data_info, f, indent=2)
        
        logger.info(f"Experiment results saved to: {output_path}")
        return str(output_path)


class ExperimentRunner:
    """
    High-level interface for running complete training experiments.
    """
    
    @staticmethod
    def run_baseline_experiment(data_path: str,
                              models: List[BaseModel],
                              experiment_name: str = "baseline") -> ModelTrainer:
        """
        Run a baseline experiment with default configuration.
        
        Args:
            data_path: Path to data file
            models: List of models to train
            experiment_name: Name for the experiment
            
        Returns:
            Trained ModelTrainer instance
        """
        config = TrainingConfig(
            experiment_name=experiment_name,
            feature_set='baseline',
            feature_selection=False,
            hyperparameter_tuning=False
        )
        
        trainer = ModelTrainer(config)
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(data_path)
        
        # Train models
        model_tuples = [(model, {}) for model in models]
        trainer.train_multiple_models(
            model_tuples, X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Save results
        trainer.save_experiment_results()
        
        return trainer
    
    @staticmethod
    def run_full_experiment(data_path: str,
                          models: List[Tuple[BaseModel, Dict[str, Any]]],
                          experiment_name: str = "full_experiment") -> ModelTrainer:
        """
        Run a full experiment with feature engineering and hyperparameter tuning.
        
        Args:
            data_path: Path to data file
            models: List of (model, hyperparameter_space) tuples
            experiment_name: Name for the experiment
            
        Returns:
            Trained ModelTrainer instance
        """
        config = TrainingConfig(
            experiment_name=experiment_name,
            feature_set='extended',
            feature_selection=True,
            hyperparameter_tuning=True
        )
        
        trainer = ModelTrainer(config)
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(data_path)
        
        # Train models
        trainer.train_multiple_models(
            models, X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Save results
        trainer.save_experiment_results()
        
        return trainer


# Utility functions for common training workflows

def create_training_config(**kwargs) -> TrainingConfig:
    """
    Create training configuration with custom parameters.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        TrainingConfig object
    """
    return TrainingConfig(**kwargs)


def load_experiment_results(experiment_path: str) -> Dict[str, Any]:
    """
    Load saved experiment results.
    
    Args:
        experiment_path: Path to experiment directory
        
    Returns:
        Dictionary with experiment data
    """
    experiment_path = Path(experiment_path)
    
    results = {}
    
    # Load training summary
    summary_file = experiment_path / "training_summary.csv"
    if summary_file.exists():
        results['summary'] = pd.read_csv(summary_file)
    
    # Load detailed results
    detailed_file = experiment_path / "detailed_results.json"
    if detailed_file.exists():
        with open(detailed_file, 'r') as f:
            results['detailed'] = json.load(f)
    
    # Load configuration
    config_file = experiment_path / "experiment_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            results['config'] = json.load(f)
    
    # Load data info
    data_info_file = experiment_path / "data_info.json"
    if data_info_file.exists():
        with open(data_info_file, 'r') as f:
            results['data_info'] = json.load(f)
    
    return results


# Example usage and testing
def test_model_trainer():
    """Test function for ModelTrainer."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    
    print("Testing ModelTrainer...")
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Save sample data
    import tempfile
    import pandas as pd
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df = pd.DataFrame(X)
        df['target'] = y
        df.to_csv(f.name, index=False)
        temp_data_path = f.name
    
    try:
        # Create models with hyperparameter spaces
        rf_model = RandomForestClassifier(random_state=42)
        rf_params = {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        
        svm_model = SVC(random_state=42, probability=True)
        svm_params = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear']
        }
        
        models = [
            (BaseModel().set_model(rf_model), rf_params),
            (BaseModel().set_model(svm_model), svm_params)
        ]
        
        # Run experiment
        trainer = ExperimentRunner.run_full_experiment(
            temp_data_path, models, "test_experiment"
        )
        
        # Check results
        assert len(trainer.training_results) > 0
        assert trainer.get_best_model() is not None
        
        summary = trainer.get_training_summary()
        assert not summary.empty
        
        print("All tests passed!")
        
    finally:
        # Cleanup
        os.unlink(temp_data_path)


if __name__ == "__main__":
    test_model_trainer()
