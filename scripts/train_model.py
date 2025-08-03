#!/usr/bin/env python3
"""
Main Training Script for AI4I Predictive Maintenance

This script orchestrates the complete machine learning pipeline for predictive maintenance:
- Data loading and preprocessing
- Feature engineering and selection
- Model training with multiple algorithms
- Hyperparameter optimization
- Model evaluation and comparison
- Model persistence and artifact management
- Comprehensive reporting and visualization

This script serves as the main entry point for training predictive maintenance models
in production environments.

Usage:
    python scripts/train_model.py --config config.yaml
    python scripts/train_model.py --data data/raw/ai4i2020.csv --target "Machine failure"
    python scripts/train_model.py --quick-run  # For testing with reduced dataset

Author: AI4I Project Team
Created: August 2025
"""

import argparse
import logging
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Data handling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import Config
from src.utils.data_preprocessing import preprocess_ai4i_data, DataPreprocessor
from src.utils.helpers import save_json, load_json, ensure_directory

from src.features.feature_engineering import FeatureEngineer
from src.features.feature_selection import FeatureSelector

from src.models.model_trainer import ModelTrainer, TrainingConfig
from src.models.model_evaluator import ModelEvaluator, EvaluationConfig
from src.models.traditional_models import TraditionalModelFactory
from src.models.neural_networks import NeuralNetworkFactory
from src.models.model_utils import ModelPersistence, save_model, load_model

from src.visualization.model_plots import ModelPlotter, compare_models
from src.visualization.eda_plots import EDAPlotter


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AI4I Predictive Maintenance Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--data', 
        type=str, 
        default='data/raw/ai4i2020.csv',
        help='Path to the AI4I dataset'
    )
    
    parser.add_argument(
        '--target', 
        type=str, 
        default='Machine failure',
        help='Target column name for prediction'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='models',
        help='Directory to save trained models'
    )
    
    # Training configuration
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--models', 
        nargs='+', 
        default=['random_forest', 'xgboost', 'svm', 'mlp'],
        help='Models to train (space-separated)'
    )
    
    parser.add_argument(
        '--feature-set', 
        type=str, 
        default='extended',
        choices=['basic', 'extended', 'comprehensive'],
        help='Feature engineering complexity level'
    )
    
    # Execution options
    parser.add_argument(
        '--quick-run', 
        action='store_true',
        help='Quick run with reduced dataset for testing'
    )
    
    parser.add_argument(
        '--no-hyperopt', 
        action='store_true',
        help='Skip hyperparameter optimization'
    )
    
    parser.add_argument(
        '--no-neural', 
        action='store_true',
        help='Skip neural network models'
    )
    
    parser.add_argument(
        '--no-plots', 
        action='store_true',
        help='Skip visualization generation'
    )
    
    # Logging
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def load_configuration(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or return defaults."""
    
    default_config = {
        'data': {
            'test_size': 0.2,
            'val_size': 0.2,
            'stratify': True,
            'preprocessing': {
                'handle_missing': True,
                'encode_categorical': True,
                'scale_features': True,
                'remove_outliers': False
            }
        },
        'features': {
            'engineering': {
                'interaction_features': True,
                'polynomial_features': False,
                'statistical_features': True,
                'domain_features': True
            },
            'selection': {
                'method': 'mutual_info',
                'max_features': 50,
                'remove_low_variance': True
            }
        },
        'training': {
            'cross_validation': {
                'cv_folds': 5,
                'scoring': 'f1_weighted'
            },
            'hyperparameter_optimization': {
                'enabled': True,
                'n_trials': 100,
                'timeout': 3600  # 1 hour
            },
            'early_stopping': {
                'enabled': True,
                'patience': 10
            }
        },
        'models': {
            'traditional': {
                'random_forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                },
                'xgboost': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'neural': {
                'mlp': {
                    'hidden_layers': [[128, 64], [256, 128, 64]],
                    'learning_rate': [0.001, 0.01],
                    'dropout_rate': [0.2, 0.3, 0.5]
                }
            }
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'],
            'business_metrics': True,
            'feature_importance': True,
            'learning_curves': True
        },
        'output': {
            'save_models': True,
            'save_predictions': True,
            'save_reports': True,
            'generate_plots': True
        }
    }
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Merge configurations (user config overrides defaults)
        def deep_merge(default: dict, user: dict) -> dict:
            for key, value in user.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    default[key] = deep_merge(default[key], value)
                else:
                    default[key] = value
            return default
        
        return deep_merge(default_config, user_config)
    
    return default_config


def setup_training_environment(args: argparse.Namespace) -> Tuple[logging.Logger, Dict[str, Any]]:
    """Setup logging and configuration for training."""
    
    # Setup logging
    logger = setup_logger(
        name='train_model',
        level=args.log_level,
        log_file=f'logs/training_{int(time.time())}.log'
    )
    
    # Load configuration
    config = load_configuration(args.config)
    
    # Override config with command line arguments
    if args.quick_run:
        config['training']['hyperparameter_optimization']['enabled'] = False
        config['training']['hyperparameter_optimization']['n_trials'] = 10
        config['data']['test_size'] = 0.3
        config['data']['val_size'] = 0.3
    
    if args.no_hyperopt:
        config['training']['hyperparameter_optimization']['enabled'] = False
    
    if args.no_neural:
        config['models'].pop('neural', None)
    
    if args.no_plots:
        config['output']['generate_plots'] = False
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directories
    ensure_directory(args.output_dir)
    ensure_directory('reports/training')
    ensure_directory('reports/figures')
    ensure_directory('logs')
    
    logger.info("Training environment setup completed")
    logger.info(f"Configuration: {config}")
    
    return logger, config


def load_and_preprocess_data(data_path: str, 
                           target_col: str, 
                           config: Dict[str, Any], 
                           logger: logging.Logger,
                           quick_run: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the AI4I dataset."""
    
    logger.info(f"Loading data from {data_path}")
    
    try:
        # Load raw data
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully: {df.shape}")
        
        # Quick run - sample data
        if quick_run:
            df = df.sample(n=min(1000, len(df)), random_state=42)
            logger.info(f"Quick run mode: Using sample of {len(df)} rows")
        
        # Check target column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        logger.info(f"Features: {X.shape[1]}, Target: {target_col}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        # Preprocessing
        preprocessor = DataPreprocessor()
        
        preprocessing_config = config['data']['preprocessing']
        
        if preprocessing_config.get('handle_missing', True):
            X = preprocessor.handle_missing_values(X)
            logger.info("Missing values handled")
        
        if preprocessing_config.get('encode_categorical', True):
            X = preprocessor.encode_categorical_features(X)
            logger.info("Categorical features encoded")
        
        if preprocessing_config.get('scale_features', True):
            X = preprocessor.scale_features(X)
            logger.info("Features scaled")
        
        logger.info(f"Preprocessing completed. Final shape: {X.shape}")
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error in data loading/preprocessing: {e}")
        raise


def engineer_and_select_features(X: pd.DataFrame, 
                                y: pd.Series, 
                                config: Dict[str, Any], 
                                logger: logging.Logger,
                                feature_set: str = 'extended') -> Tuple[pd.DataFrame, List[str]]:
    """Engineer and select features for model training."""
    
    logger.info("Starting feature engineering...")
    
    try:
        # Feature Engineering
        engineer = FeatureEngineer()
        
        engineering_config = config['features']['engineering']
        
        # Create feature set based on complexity level
        if feature_set == 'basic':
            X_engineered = engineer.create_basic_features(X)
        elif feature_set == 'extended':
            X_engineered = engineer.get_feature_set(X, 'extended')
        else:  # comprehensive
            X_engineered = engineer.get_feature_set(X, 'comprehensive')
        
        logger.info(f"Feature engineering completed: {X_engineered.shape[1]} features")
        
        # Feature Selection
        selector = FeatureSelector()
        selection_config = config['features']['selection']
        
        # Select best features
        selected_features = selector.select_best_features(
            X_engineered, y,
            method=selection_config.get('method', 'mutual_info'),
            max_features=selection_config.get('max_features', 50)
        )
        
        X_selected = X_engineered[selected_features]
        
        logger.info(f"Feature selection completed: {len(selected_features)} features selected")
        logger.info(f"Selected features: {selected_features[:10]}...")  # Show first 10
        
        return X_selected, selected_features
        
    except Exception as e:
        logger.error(f"Error in feature engineering/selection: {e}")
        raise


def create_model_suite(config: Dict[str, Any], 
                      logger: logging.Logger,
                      model_names: List[str]) -> Dict[str, Any]:
    """Create suite of models for training."""
    
    logger.info("Creating model suite...")
    
    models = {}
    
    try:
        # Traditional Models
        traditional_factory = TraditionalModelFactory()
        
        for model_name in model_names:
            if model_name in ['random_forest', 'rf']:
                models['Random Forest'] = traditional_factory.create_model('random_forest_classifier')
            elif model_name in ['xgboost', 'xgb']:
                models['XGBoost'] = traditional_factory.create_model('xgboost_classifier')
            elif model_name in ['svm']:
                models['SVM'] = traditional_factory.create_model('svm_classifier')
            elif model_name in ['logistic', 'lr']:
                models['Logistic Regression'] = traditional_factory.create_model('logistic_regression')
            elif model_name in ['lightgbm', 'lgb']:
                models['LightGBM'] = traditional_factory.create_model('lightgbm_classifier')
        
        # Neural Networks (if not disabled)
        if not config.get('no_neural', False):
            try:
                neural_factory = NeuralNetworkFactory()
                
                for model_name in model_names:
                    if model_name in ['mlp', 'neural', 'nn']:
                        models['Neural Network'] = neural_factory.create_model('mlp_classifier')
                    elif model_name in ['lstm']:
                        models['LSTM'] = neural_factory.create_model('lstm_classifier')
                        
            except ImportError:
                logger.warning("Neural networks not available (TensorFlow not installed)")
        
        logger.info(f"Model suite created: {list(models.keys())}")
        return models
        
    except Exception as e:
        logger.error(f"Error creating model suite: {e}")
        raise


def train_models(models: Dict[str, Any], 
                X_train: pd.DataFrame, 
                y_train: pd.Series,
                X_val: pd.DataFrame, 
                y_val: pd.Series,
                config: Dict[str, Any], 
                logger: logging.Logger) -> Dict[str, Any]:
    """Train all models with the given data."""
    
    logger.info("Starting model training...")
    
    training_results = {}
    
    try:
        # Setup trainer
        training_config = TrainingConfig(
            cv_folds=config['training']['cross_validation']['cv_folds'],
            scoring=config['training']['cross_validation']['scoring'],
            hyperparameter_optimization=config['training']['hyperparameter_optimization']['enabled'],
            n_trials=config['training']['hyperparameter_optimization']['n_trials'],
            timeout=config['training']['hyperparameter_optimization']['timeout']
        )
        
        trainer = ModelTrainer(config=training_config)
        
        # Train each model
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            start_time = time.time()
            
            try:
                # Train model
                result = trainer.train_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val
                )
                
                training_time = time.time() - start_time
                result['training_time'] = training_time
                result['model_name'] = model_name
                
                training_results[model_name] = result
                
                logger.info(f"{model_name} training completed in {training_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        logger.info(f"Model training completed: {len(training_results)} models trained")
        return training_results
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise


def evaluate_models(training_results: Dict[str, Any],
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   config: Dict[str, Any],
                   logger: logging.Logger) -> Dict[str, Any]:
    """Evaluate all trained models."""
    
    logger.info("Starting model evaluation...")
    
    evaluation_results = {}
    
    try:
        # Setup evaluator
        evaluation_config = EvaluationConfig(
            metrics=config['evaluation']['metrics'],
            business_analysis=config['evaluation']['business_metrics'],
            feature_importance=config['evaluation']['feature_importance']
        )
        
        evaluator = ModelEvaluator(config=evaluation_config)
        
        # Evaluate each model
        for model_name, training_result in training_results.items():
            logger.info(f"Evaluating {model_name}...")
            
            try:
                model = training_result['model']
                
                # Evaluate model
                eval_result = evaluator.evaluate_model(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    model_name=model_name
                )
                
                # Combine training and evaluation results
                combined_result = {
                    **training_result,
                    **eval_result.__dict__
                }
                
                evaluation_results[model_name] = combined_result
                
                logger.info(f"{model_name} evaluation completed")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        logger.info(f"Model evaluation completed: {len(evaluation_results)} models evaluated")
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise


def save_models_and_artifacts(evaluation_results: Dict[str, Any],
                             selected_features: List[str],
                             config: Dict[str, Any],
                             output_dir: str,
                             logger: logging.Logger) -> None:
    """Save trained models and related artifacts."""
    
    logger.info("Saving models and artifacts...")
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save each model
        for model_name, results in evaluation_results.items():
            model = results['model']
            
            # Create model-specific directory
            model_dir = output_path / model_name.replace(' ', '_').lower()
            model_dir.mkdir(exist_ok=True)
            
            # Save model
            model_file = model_dir / 'model.pkl'
            save_model(model, str(model_file))
            
            # Save model metadata
            metadata = {
                'model_name': model_name,
                'training_time': results.get('training_time', None),
                'performance_metrics': {
                    'accuracy': results.get('accuracy', None),
                    'precision': results.get('precision', None),
                    'recall': results.get('recall', None),
                    'f1_score': results.get('f1_score', None),
                    'auc_roc': results.get('auc_roc', None)
                },
                'selected_features': selected_features,
                'feature_count': len(selected_features),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            save_json(metadata, model_dir / 'metadata.json')
            
            logger.info(f"Saved {model_name} to {model_dir}")
        
        # Save overall training summary
        summary = {
            'training_summary': {
                'models_trained': list(evaluation_results.keys()),
                'best_model': max(evaluation_results.keys(), 
                                key=lambda k: evaluation_results[k].get('f1_score', 0)),
                'total_features': len(selected_features),
                'configuration': config,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        save_json(summary, output_path / 'training_summary.json')
        
        logger.info("Models and artifacts saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        raise


def generate_visualizations(evaluation_results: Dict[str, Any],
                          X_test: pd.DataFrame,
                          y_test: pd.Series,
                          selected_features: List[str],
                          config: Dict[str, Any],
                          logger: logging.Logger) -> None:
    """Generate training and evaluation visualizations."""
    
    if not config['output'].get('generate_plots', True):
        logger.info("Visualization generation skipped")
        return
    
    logger.info("Generating visualizations...")
    
    try:
        plotter = ModelPlotter(save_plots=True, output_dir='reports/figures')
        
        # Model comparison
        comparison_data = {}
        for model_name, results in evaluation_results.items():
            comparison_data[model_name] = {
                'accuracy': results.get('accuracy', 0),
                'precision': results.get('precision', 0),
                'recall': results.get('recall', 0),
                'f1_score': results.get('f1_score', 0),
                'auc_roc': results.get('auc_roc', 0)
            }
        
        plotter.plot_model_comparison(comparison_data, 'classification', 
                                     title="AI4I Model Comparison Results")
        
        # Individual model performance plots
        for model_name, results in evaluation_results.items():
            if 'y_pred' in results and 'y_proba' in results:
                plotter.plot_classification_performance(
                    y_test.values, 
                    results['y_pred'],
                    results.get('y_proba'),
                    model_name
                )
                
                # Feature importance if available
                if 'feature_importance' in results:
                    plotter.plot_feature_importance(
                        selected_features,
                        results['feature_importance'],
                        model_name
                    )
        
        logger.info("Visualizations generated successfully")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")


def generate_training_report(evaluation_results: Dict[str, Any],
                           config: Dict[str, Any],
                           training_duration: float,
                           logger: logging.Logger) -> None:
    """Generate comprehensive training report."""
    
    logger.info("Generating training report...")
    
    try:
        # Find best model
        best_model_name = max(evaluation_results.keys(), 
                            key=lambda k: evaluation_results[k].get('f1_score', 0))
        best_results = evaluation_results[best_model_name]
        
        # Generate report
        report = f"""
AI4I PREDICTIVE MAINTENANCE - TRAINING REPORT
============================================

Training Summary:
-----------------
• Total Training Time: {training_duration:.2f} seconds
• Models Trained: {len(evaluation_results)}
• Best Model: {best_model_name}
• Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}

Best Model Performance:
----------------------
• Accuracy: {best_results.get('accuracy', 0):.3f}
• Precision: {best_results.get('precision', 0):.3f}
• Recall: {best_results.get('recall', 0):.3f}
• F1-Score: {best_results.get('f1_score', 0):.3f}
• AUC-ROC: {best_results.get('auc_roc', 0):.3f}

All Models Comparison:
---------------------
"""
        
        for model_name, results in evaluation_results.items():
            report += f"• {model_name:20} | F1: {results.get('f1_score', 0):.3f} | "
            report += f"Accuracy: {results.get('accuracy', 0):.3f}\n"
        
        report += f"""

Configuration Used:
------------------
• Feature Set: {config.get('feature_set', 'extended')}
• Hyperparameter Optimization: {config['training']['hyperparameter_optimization']['enabled']}
• Cross-Validation Folds: {config['training']['cross_validation']['cv_folds']}

Recommendations:
---------------
"""
        
        # Add recommendations based on results
        best_f1 = best_results.get('f1_score', 0)
        if best_f1 > 0.9:
            report += "• Excellent performance achieved! Ready for production deployment.\n"
        elif best_f1 > 0.8:
            report += "• Good performance. Consider additional feature engineering for improvement.\n"
        else:
            report += "• Performance below target. Recommend data quality review and feature engineering.\n"
        
        # Save report
        with open('reports/training/training_report.txt', 'w') as f:
            f.write(report)
        
        # Also log to console
        logger.info("Training Report Generated:")
        logger.info(report)
        
    except Exception as e:
        logger.error(f"Error generating training report: {e}")


def main():
    """Main training pipeline."""
    
    print("🚀 AI4I Predictive Maintenance - Model Training Pipeline")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    logger, config = setup_training_environment(args)
    
    training_start_time = time.time()
    
    try:
        # 1. Data Loading and Preprocessing
        print("\n📊 Step 1: Data Loading and Preprocessing")
        X, y = load_and_preprocess_data(args.data, args.target, config, logger, args.quick_run)
        
        # 2. Feature Engineering and Selection
        print("\n🔧 Step 2: Feature Engineering and Selection")
        X_features, selected_features = engineer_and_select_features(X, y, config, logger, args.feature_set)
        
        # 3. Train/Validation/Test Split
        print("\n📝 Step 3: Data Splitting")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_features, y, 
            test_size=config['data']['test_size'], 
            stratify=y if config['data']['stratify'] else None,
            random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=config['data']['val_size']/(1-config['data']['test_size']),
            stratify=y_temp if config['data']['stratify'] else None,
            random_state=42
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # 4. Model Creation
        print("\n🤖 Step 4: Model Creation")
        models = create_model_suite(config, logger, args.models)
        
        # 5. Model Training
        print("\n⚡ Step 5: Model Training")
        training_results = train_models(models, X_train, y_train, X_val, y_val, config, logger)
        
        # 6. Model Evaluation
        print("\n📈 Step 6: Model Evaluation")
        evaluation_results = evaluate_models(training_results, X_test, y_test, config, logger)
        
        # 7. Save Models and Artifacts
        print("\n💾 Step 7: Saving Models and Artifacts")
        save_models_and_artifacts(evaluation_results, selected_features, config, args.output_dir, logger)
        
        # 8. Generate Visualizations
        print("\n📊 Step 8: Generating Visualizations")
        generate_visualizations(evaluation_results, X_test, y_test, selected_features, config, logger)
        
        # 9. Generate Report
        print("\n📋 Step 9: Generating Training Report")
        training_duration = time.time() - training_start_time
        generate_training_report(evaluation_results, config, training_duration, logger)
        
        # Success summary
        best_model = max(evaluation_results.keys(), 
                        key=lambda k: evaluation_results[k].get('f1_score', 0))
        best_f1 = evaluation_results[best_model].get('f1_score', 0)
        
        print(f"\n✅ Training Pipeline Completed Successfully!")
        print(f"⏱️  Total Time: {training_duration:.2f} seconds")
        print(f"🏆 Best Model: {best_model} (F1-Score: {best_f1:.3f})")
        print(f"📁 Models saved to: {args.output_dir}")
        print(f"📊 Reports saved to: reports/")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        print(f"\n❌ Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
