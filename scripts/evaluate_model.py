#!/usr/bin/env python3
"""
Model Evaluation Script for AI4I Predictive Maintenance

This script provides comprehensive model evaluation capabilities for trained AI4I models:
- Load and evaluate pre-trained models
- Detailed performance analysis with multiple metrics
- Business impact assessment and ROI calculations
- Model comparison and benchmarking
- Comprehensive reporting and visualization
- Model deployment readiness assessment

This script can be used standalone for evaluating existing models or as part of the
continuous evaluation pipeline for model monitoring in production.

Usage:
    python scripts/evaluate_model.py --model models/random_forest/model.pkl
    python scripts/evaluate_model.py --model-dir models/ --compare-all
    python scripts/evaluate_model.py --test-data data/test/test_set.csv
    python scripts/evaluate_model.py --production-metrics --business-analysis

Author: AI4I Project Team
Created: August 2025
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Data handling
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, average_precision_score
)

# Our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import Config
from src.utils.data_preprocessing import DataPreprocessor
from src.utils.helpers import load_json, save_json, ensure_directory

from src.models.model_evaluator import ModelEvaluator, EvaluationConfig, EvaluationResult
from src.models.model_utils import load_model, ModelPersistence
from src.models.base_model import BaseModel

from src.visualization.model_plots import ModelPlotter
from src.visualization.report_plots import BusinessReportGenerator


class ModelEvaluationPipeline:
    """
    Comprehensive model evaluation pipeline for AI4I predictive maintenance models.
    
    This class orchestrates the complete evaluation process including:
    - Model loading and validation
    - Test data preparation
    - Performance metric calculation
    - Business impact analysis
    - Visualization generation
    - Report creation
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the evaluation pipeline.
        
        Args:
            config: Configuration dictionary for evaluation parameters
            logger: Logger instance for tracking evaluation progress
        """
        self.config = config or self._get_default_config()
        self.logger = logger or self._setup_default_logger()
        
        # Initialize components
        self.evaluator = ModelEvaluator(
            config=EvaluationConfig(
                metrics=self.config['evaluation']['metrics'],
                business_analysis=self.config['evaluation']['business_analysis'],
                feature_importance=self.config['evaluation']['feature_importance']
            )
        )
        
        self.plotter = ModelPlotter(
            save_plots=self.config['output']['save_plots'],
            output_dir=self.config['output']['plot_dir']
        )
        
        # Store evaluation results
        self.evaluation_results = {}
        self.model_metadata = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for evaluation."""
        return {
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'],
                'business_analysis': True,
                'feature_importance': True,
                'threshold_analysis': True,
                'calibration_analysis': True
            },
            'business': {
                'cost_per_failure': 10000,  # Cost of unplanned failure
                'cost_per_maintenance': 1000,  # Cost of planned maintenance
                'production_value_per_hour': 5000,  # Revenue impact per hour
                'maintenance_duration_hours': 4  # Hours for maintenance
            },
            'output': {
                'save_plots': True,
                'save_reports': True,
                'plot_dir': 'reports/figures/evaluation',
                'report_dir': 'reports/evaluation',
                'detailed_analysis': True
            },
            'thresholds': {
                'accuracy_minimum': 0.85,
                'precision_minimum': 0.80,
                'recall_minimum': 0.85,
                'f1_minimum': 0.82,
                'production_ready_threshold': 0.85
            }
        }
    
    def _setup_default_logger(self) -> logging.Logger:
        """Setup default logger for evaluation."""
        return setup_logger(
            name='model_evaluation',
            level='INFO',
            log_file=f'logs/evaluation_{int(time.time())}.log'
        )
    
    def load_model_with_metadata(self, model_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load model and its associated metadata.
        
        Args:
            model_path: Path to the model file or model directory
            
        Returns:
            Tuple of (model, metadata)
        """
        model_path = Path(model_path)
        
        try:
            # Handle directory path (look for model.pkl)
            if model_path.is_dir():
                model_file = model_path / 'model.pkl'
                metadata_file = model_path / 'metadata.json'
            else:
                model_file = model_path
                metadata_file = model_path.parent / 'metadata.json'
            
            # Load model
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            model = load_model(str(model_file))
            self.logger.info(f"Model loaded from {model_file}")
            
            # Load metadata if available
            metadata = {}
            if metadata_file.exists():
                metadata = load_json(metadata_file)
                self.logger.info(f"Metadata loaded from {metadata_file}")
            else:
                self.logger.warning(f"No metadata found at {metadata_file}")
                metadata = {
                    'model_name': model_path.name,
                    'timestamp': 'unknown',
                    'feature_count': 'unknown'
                }
            
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def load_test_data(self, 
                      test_data_path: str,
                      target_column: str = 'Machine failure',
                      feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare test data for evaluation.
        
        Args:
            test_data_path: Path to test data CSV file
            target_column: Name of target column
            feature_columns: List of feature columns to use (if None, use all except target)
            
        Returns:
            Tuple of (X_test, y_test)
        """
        try:
            # Load data
            df = pd.read_csv(test_data_path)
            self.logger.info(f"Test data loaded: {df.shape}")
            
            # Check target column
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in test data")
            
            # Separate features and target
            if feature_columns:
                # Use specified features
                missing_features = set(feature_columns) - set(df.columns)
                if missing_features:
                    raise ValueError(f"Features not found in data: {missing_features}")
                X_test = df[feature_columns]
            else:
                # Use all columns except target
                X_test = df.drop(columns=[target_column])
            
            y_test = df[target_column]
            
            self.logger.info(f"Test data prepared - Features: {X_test.shape[1]}, Samples: {len(y_test)}")
            self.logger.info(f"Target distribution:\n{y_test.value_counts()}")
            
            return X_test, y_test
            
        except Exception as e:
            self.logger.error(f"Error loading test data: {e}")
            raise
    
    def evaluate_single_model(self, 
                             model: Any,
                             X_test: pd.DataFrame,
                             y_test: pd.Series,
                             model_name: str,
                             metadata: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """
        Evaluate a single model on test data.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test target
            model_name: Name of the model for reporting
            metadata: Model metadata for additional analysis
            
        Returns:
            Evaluation results
        """
        self.logger.info(f"Evaluating model: {model_name}")
        
        try:
            # Perform evaluation
            result = self.evaluator.evaluate_model(
                model=model,
                X_test=X_test,
                y_test=y_test,
                model_name=model_name
            )
            
            # Add metadata to results
            if metadata:
                result.model_metadata = metadata
            
            # Business impact analysis
            if self.config['evaluation']['business_analysis']:
                business_metrics = self._calculate_business_metrics(
                    y_test, result.y_pred, result.y_proba
                )
                result.business_metrics = business_metrics
            
            # Threshold analysis
            if self.config['evaluation']['threshold_analysis']:
                threshold_analysis = self._perform_threshold_analysis(
                    y_test, result.y_proba
                )
                result.threshold_analysis = threshold_analysis
            
            # Production readiness assessment
            result.production_ready = self._assess_production_readiness(result)
            
            self.logger.info(f"Model {model_name} evaluation completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_name}: {e}")
            raise
    
    def _calculate_business_metrics(self,
                                  y_true: pd.Series,
                                  y_pred: np.ndarray,
                                  y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate business impact metrics."""
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Business costs
        cost_per_failure = self.config['business']['cost_per_failure']
        cost_per_maintenance = self.config['business']['cost_per_maintenance']
        production_value = self.config['business']['production_value_per_hour']
        maintenance_hours = self.config['business']['maintenance_duration_hours']
        
        # Calculate business metrics
        # Cost of false negatives (missed failures)
        cost_missed_failures = fn * cost_per_failure
        
        # Cost of false positives (unnecessary maintenance)
        cost_unnecessary_maintenance = fp * cost_per_maintenance
        
        # Production loss from maintenance
        production_loss = (tp + fp) * maintenance_hours * production_value
        
        # Total cost
        total_cost = cost_missed_failures + cost_unnecessary_maintenance + production_loss
        
        # Savings compared to no prediction (all failures unplanned)
        baseline_cost = len(y_true) * cost_per_failure
        cost_savings = baseline_cost - total_cost
        cost_savings_percentage = (cost_savings / baseline_cost) * 100 if baseline_cost > 0 else 0
        
        # ROI calculation
        roi = (cost_savings / baseline_cost) * 100 if baseline_cost > 0 else 0
        
        return {
            'cost_missed_failures': cost_missed_failures,
            'cost_unnecessary_maintenance': cost_unnecessary_maintenance,
            'production_loss': production_loss,
            'total_cost': total_cost,
            'cost_savings': cost_savings,
            'cost_savings_percentage': cost_savings_percentage,
            'roi_percentage': roi,
            'baseline_cost': baseline_cost
        }
    
    def _perform_threshold_analysis(self,
                                  y_true: pd.Series,
                                  y_proba: np.ndarray) -> Dict[str, Any]:
        """Perform threshold analysis for optimal decision boundary."""
        
        if y_proba is None:
            return {}
        
        # Use probability of positive class
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_proba)
        
        # Calculate ROC curve
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_proba)
        
        # Find optimal threshold (max F1-score)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1])
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds_pr[optimal_idx]
        
        # Calculate metrics at optimal threshold
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
        optimal_metrics = {
            'accuracy': accuracy_score(y_true, y_pred_optimal),
            'precision': precision_score(y_true, y_pred_optimal),
            'recall': recall_score(y_true, y_pred_optimal),
            'f1_score': f1_score(y_true, y_pred_optimal)
        }
        
        return {
            'optimal_threshold': optimal_threshold,
            'optimal_f1_score': f1_scores[optimal_idx],
            'optimal_metrics': optimal_metrics,
            'precision_recall_curve': {
                'precisions': precisions.tolist(),
                'recalls': recalls.tolist(),
                'thresholds': thresholds_pr.tolist()
            },
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds_roc.tolist()
            }
        }
    
    def _assess_production_readiness(self, result: EvaluationResult) -> Dict[str, Any]:
        """Assess if model is ready for production deployment."""
        
        thresholds = self.config['thresholds']
        
        # Check individual metrics
        checks = {
            'accuracy_check': result.accuracy >= thresholds['accuracy_minimum'],
            'precision_check': result.precision >= thresholds['precision_minimum'],
            'recall_check': result.recall >= thresholds['recall_minimum'],
            'f1_check': result.f1_score >= thresholds['f1_minimum']
        }
        
        # Overall production readiness
        overall_ready = all(checks.values())
        
        # Confidence score (weighted average of metrics vs thresholds)
        weights = {'accuracy': 0.2, 'precision': 0.3, 'recall': 0.3, 'f1_score': 0.2}
        confidence_score = sum(
            weights[metric] * min(getattr(result, metric) / thresholds[f'{metric}_minimum'], 1.0)
            for metric in weights.keys()
        )
        
        # Risk assessment
        risk_level = 'LOW' if confidence_score >= 0.95 else 'MEDIUM' if confidence_score >= 0.85 else 'HIGH'
        
        return {
            'ready_for_production': overall_ready,
            'confidence_score': confidence_score,
            'risk_level': risk_level,
            'individual_checks': checks,
            'recommendations': self._generate_recommendations(result, checks)
        }
    
    def _generate_recommendations(self, 
                                result: EvaluationResult,
                                checks: Dict[str, bool]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        
        recommendations = []
        
        if not checks['accuracy_check']:
            recommendations.append("Improve overall accuracy through better feature engineering or model tuning")
        
        if not checks['precision_check']:
            recommendations.append("Reduce false positives to minimize unnecessary maintenance costs")
        
        if not checks['recall_check']:
            recommendations.append("Improve failure detection to reduce missed critical failures")
        
        if not checks['f1_check']:
            recommendations.append("Balance precision and recall for optimal F1-score")
        
        # Business-specific recommendations
        if hasattr(result, 'business_metrics'):
            roi = result.business_metrics.get('roi_percentage', 0)
            if roi < 50:
                recommendations.append("Business ROI below target - consider cost-benefit optimization")
        
        if not recommendations:
            recommendations.append("Model performance meets production standards - ready for deployment")
        
        return recommendations
    
    def compare_models(self, model_results: Dict[str, EvaluationResult]) -> Dict[str, Any]:
        """
        Compare multiple models and identify the best performer.
        
        Args:
            model_results: Dictionary mapping model names to evaluation results
            
        Returns:
            Comparison analysis
        """
        self.logger.info(f"Comparing {len(model_results)} models")
        
        # Create comparison matrix
        comparison_data = {}
        for name, result in model_results.items():
            comparison_data[name] = {
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'auc_roc': result.auc_roc,
                'production_ready': result.production_ready.get('ready_for_production', False),
                'confidence_score': result.production_ready.get('confidence_score', 0),
                'roi_percentage': result.business_metrics.get('roi_percentage', 0) if hasattr(result, 'business_metrics') else 0
            }
        
        # Find best models
        best_f1 = max(comparison_data.keys(), key=lambda k: comparison_data[k]['f1_score'])
        best_roi = max(comparison_data.keys(), key=lambda k: comparison_data[k]['roi_percentage'])
        best_overall = max(comparison_data.keys(), key=lambda k: comparison_data[k]['confidence_score'])
        
        # Model rankings
        rankings = {
            'by_f1_score': sorted(comparison_data.keys(), 
                                key=lambda k: comparison_data[k]['f1_score'], reverse=True),
            'by_roi': sorted(comparison_data.keys(), 
                           key=lambda k: comparison_data[k]['roi_percentage'], reverse=True),
            'by_production_readiness': sorted(comparison_data.keys(), 
                                            key=lambda k: comparison_data[k]['confidence_score'], reverse=True)
        }
        
        return {
            'comparison_matrix': comparison_data,
            'best_models': {
                'best_f1_score': best_f1,
                'best_roi': best_roi,
                'best_overall': best_overall
            },
            'rankings': rankings,
            'production_ready_models': [name for name, data in comparison_data.items() 
                                      if data['production_ready']],
            'summary_stats': {
                'total_models': len(model_results),
                'production_ready_count': sum(data['production_ready'] for data in comparison_data.values()),
                'average_f1_score': np.mean([data['f1_score'] for data in comparison_data.values()]),
                'average_roi': np.mean([data['roi_percentage'] for data in comparison_data.values()])
            }
        }
    
    def generate_visualizations(self, model_results: Dict[str, EvaluationResult]) -> None:
        """Generate comprehensive visualizations for evaluation results."""
        
        if not self.config['output']['save_plots']:
            self.logger.info("Visualization generation skipped")
            return
        
        self.logger.info("Generating evaluation visualizations...")
        
        try:
            # Ensure output directory exists
            ensure_directory(self.config['output']['plot_dir'])
            
            # Model comparison visualization
            if len(model_results) > 1:
                comparison_data = {}
                for name, result in model_results.items():
                    comparison_data[name] = {
                        'accuracy': result.accuracy,
                        'precision': result.precision,
                        'recall': result.recall,
                        'f1_score': result.f1_score,
                        'auc_roc': result.auc_roc
                    }
                
                self.plotter.plot_model_comparison(
                    comparison_data, 
                    'classification',
                    title="AI4I Model Evaluation Comparison"
                )
            
            # Individual model visualizations
            for model_name, result in model_results.items():
                # Performance plots
                self.plotter.plot_classification_performance(
                    result.y_true, 
                    result.y_pred,
                    result.y_proba,
                    model_name
                )
                
                # Feature importance if available
                if hasattr(result, 'feature_importance') and result.feature_importance is not None:
                    feature_names = getattr(result, 'feature_names', [f'Feature_{i}' for i in range(len(result.feature_importance))])
                    self.plotter.plot_feature_importance(
                        feature_names,
                        result.feature_importance,
                        model_name
                    )
                
                # Business metrics visualization
                if hasattr(result, 'business_metrics'):
                    self._plot_business_metrics(result.business_metrics, model_name)
            
            self.logger.info("Visualizations generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
    
    def _plot_business_metrics(self, business_metrics: Dict[str, float], model_name: str) -> None:
        """Plot business impact metrics."""
        
        try:
            import matplotlib.pyplot as plt
            
            # Cost breakdown
            costs = {
                'Missed Failures': business_metrics['cost_missed_failures'],
                'Unnecessary Maintenance': business_metrics['cost_unnecessary_maintenance'],
                'Production Loss': business_metrics['production_loss']
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Cost breakdown pie chart
            ax1.pie(costs.values(), labels=costs.keys(), autopct='%1.1f%%', startangle=90)
            ax1.set_title(f'{model_name} - Cost Breakdown')
            
            # ROI and savings bar chart
            savings_data = {
                'Cost Savings': business_metrics['cost_savings'],
                'ROI %': business_metrics['roi_percentage'] * 1000  # Scale for visibility
            }
            
            ax2.bar(savings_data.keys(), savings_data.values(), color=['green', 'blue'])
            ax2.set_title(f'{model_name} - Business Impact')
            ax2.set_ylabel('Value ($)')
            
            plt.tight_layout()
            plt.savefig(f"{self.config['output']['plot_dir']}/{model_name}_business_metrics.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Could not create business metrics plot: {e}")
    
    def generate_evaluation_report(self, 
                                 model_results: Dict[str, EvaluationResult],
                                 comparison_analysis: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive evaluation report."""
        
        self.logger.info("Generating evaluation report...")
        
        # Calculate comparison if not provided
        if comparison_analysis is None and len(model_results) > 1:
            comparison_analysis = self.compare_models(model_results)
        
        report_lines = [
            "AI4I PREDICTIVE MAINTENANCE - MODEL EVALUATION REPORT",
            "=" * 60,
            "",
            f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Models Evaluated: {len(model_results)}",
            ""
        ]
        
        # Executive Summary
        if comparison_analysis:
            best_model = comparison_analysis['best_models']['best_overall']
            production_ready = comparison_analysis['summary_stats']['production_ready_count']
            
            report_lines.extend([
                "EXECUTIVE SUMMARY",
                "-" * 20,
                f"• Best Overall Model: {best_model}",
                f"• Production-Ready Models: {production_ready}/{len(model_results)}",
                f"• Average F1-Score: {comparison_analysis['summary_stats']['average_f1_score']:.3f}",
                f"• Average ROI: {comparison_analysis['summary_stats']['average_roi']:.1f}%",
                ""
            ])
        
        # Individual Model Results
        report_lines.extend([
            "DETAILED MODEL RESULTS",
            "-" * 25
        ])
        
        for model_name, result in model_results.items():
            report_lines.extend([
                "",
                f"Model: {model_name}",
                "─" * (len(model_name) + 7),
                f"• Accuracy:     {result.accuracy:.3f}",
                f"• Precision:    {result.precision:.3f}",
                f"• Recall:       {result.recall:.3f}",
                f"• F1-Score:     {result.f1_score:.3f}",
                f"• AUC-ROC:      {result.auc_roc:.3f}",
            ])
            
            # Production readiness
            if hasattr(result, 'production_ready'):
                ready = result.production_ready['ready_for_production']
                confidence = result.production_ready['confidence_score']
                risk = result.production_ready['risk_level']
                
                report_lines.extend([
                    f"• Production Ready: {'✓ YES' if ready else '✗ NO'}",
                    f"• Confidence Score: {confidence:.3f}",
                    f"• Risk Level: {risk}",
                ])
            
            # Business metrics
            if hasattr(result, 'business_metrics'):
                roi = result.business_metrics['roi_percentage']
                savings = result.business_metrics['cost_savings']
                
                report_lines.extend([
                    f"• ROI: {roi:.1f}%",
                    f"• Cost Savings: ${savings:,.0f}",
                ])
            
            # Recommendations
            if hasattr(result, 'production_ready') and 'recommendations' in result.production_ready:
                report_lines.extend([
                    "• Recommendations:",
                    *[f"  - {rec}" for rec in result.production_ready['recommendations']]
                ])
        
        # Model Comparison
        if comparison_analysis and len(model_results) > 1:
            report_lines.extend([
                "",
                "MODEL COMPARISON & RANKINGS",
                "-" * 30,
                "",
                "F1-Score Ranking:",
                *[f"  {i+1}. {model}" for i, model in enumerate(comparison_analysis['rankings']['by_f1_score'])],
                "",
                "Business ROI Ranking:",
                *[f"  {i+1}. {model}" for i, model in enumerate(comparison_analysis['rankings']['by_roi'])],
                "",
                "Production Readiness Ranking:",
                *[f"  {i+1}. {model}" for i, model in enumerate(comparison_analysis['rankings']['by_production_readiness'])],
            ])
        
        # Deployment Recommendations
        report_lines.extend([
            "",
            "DEPLOYMENT RECOMMENDATIONS",
            "-" * 30,
        ])
        
        if comparison_analysis:
            production_ready_models = comparison_analysis['production_ready_models']
            if production_ready_models:
                best_model = comparison_analysis['best_models']['best_overall']
                report_lines.extend([
                    f"• RECOMMENDED FOR DEPLOYMENT: {best_model}",
                    f"• Alternative Options: {', '.join([m for m in production_ready_models if m != best_model])}",
                ])
            else:
                report_lines.extend([
                    "• NO MODELS MEET PRODUCTION STANDARDS",
                    "• Additional training and optimization required",
                ])
        else:
            # Single model evaluation
            model_name = list(model_results.keys())[0]
            result = model_results[model_name]
            if hasattr(result, 'production_ready') and result.production_ready['ready_for_production']:
                report_lines.append(f"• RECOMMENDED FOR DEPLOYMENT: {model_name}")
            else:
                report_lines.append("• ADDITIONAL OPTIMIZATION REQUIRED")
        
        report_lines.extend([
            "",
            "=" * 60,
            "End of Evaluation Report"
        ])
        
        return "\n".join(report_lines)
    
    def save_evaluation_results(self, 
                              model_results: Dict[str, EvaluationResult],
                              comparison_analysis: Optional[Dict[str, Any]] = None) -> None:
        """Save evaluation results and reports."""
        
        if not self.config['output']['save_reports']:
            return
        
        self.logger.info("Saving evaluation results...")
        
        try:
            # Ensure output directory exists
            ensure_directory(self.config['output']['report_dir'])
            
            # Save detailed results
            detailed_results = {}
            for model_name, result in model_results.items():
                detailed_results[model_name] = {
                    'metrics': {
                        'accuracy': result.accuracy,
                        'precision': result.precision,
                        'recall': result.recall,
                        'f1_score': result.f1_score,
                        'auc_roc': result.auc_roc
                    },
                    'production_ready': getattr(result, 'production_ready', {}),
                    'business_metrics': getattr(result, 'business_metrics', {}),
                    'threshold_analysis': getattr(result, 'threshold_analysis', {}),
                    'model_metadata': getattr(result, 'model_metadata', {})
                }
            
            # Save as JSON
            save_json(detailed_results, f"{self.config['output']['report_dir']}/detailed_results.json")
            
            # Save comparison analysis
            if comparison_analysis:
                save_json(comparison_analysis, f"{self.config['output']['report_dir']}/comparison_analysis.json")
            
            # Generate and save text report
            report_text = self.generate_evaluation_report(model_results, comparison_analysis)
            with open(f"{self.config['output']['report_dir']}/evaluation_report.txt", 'w') as f:
                f.write(report_text)
            
            self.logger.info("Evaluation results saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation results: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate AI4I Predictive Maintenance Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model specification
    parser.add_argument(
        '--model', 
        type=str,
        help='Path to single model file or model directory'
    )
    
    parser.add_argument(
        '--model-dir', 
        type=str,
        help='Directory containing multiple models to evaluate'
    )
    
    parser.add_argument(
        '--compare-all', 
        action='store_true',
        help='Compare all models in model directory'
    )
    
    # Data specification
    parser.add_argument(
        '--test-data', 
        type=str,
        help='Path to test data CSV file'
    )
    
    parser.add_argument(
        '--target-column', 
        type=str, 
        default='Machine failure',
        help='Name of target column in test data'
    )
    
    # Analysis options
    parser.add_argument(
        '--business-analysis', 
        action='store_true',
        help='Include business impact analysis'
    )
    
    parser.add_argument(
        '--production-metrics', 
        action='store_true',
        help='Include production readiness assessment'
    )
    
    parser.add_argument(
        '--threshold-analysis', 
        action='store_true',
        help='Perform threshold optimization analysis'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='reports/evaluation',
        help='Directory to save evaluation reports'
    )
    
    parser.add_argument(
        '--no-plots', 
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--no-reports', 
        action='store_true',
        help='Skip report generation'
    )
    
    # Configuration
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def discover_models(model_dir: str) -> List[str]:
    """Discover all model directories in the given path."""
    model_dir = Path(model_dir)
    model_paths = []
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Look for model.pkl files in subdirectories
    for item in model_dir.iterdir():
        if item.is_dir():
            model_file = item / 'model.pkl'
            if model_file.exists():
                model_paths.append(str(item))
    
    # Also check for direct .pkl files
    for item in model_dir.glob('*.pkl'):
        model_paths.append(str(item))
    
    return model_paths


def main():
    """Main evaluation pipeline."""
    
    print("🎯 AI4I Predictive Maintenance - Model Evaluation Pipeline")
    print("=" * 65)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger(
        name='evaluate_model',
        level=args.log_level,
        log_file=f'logs/evaluation_{int(time.time())}.log'
    )
    
    evaluation_start_time = time.time()
    
    try:
        # Validate arguments
        if not args.model and not args.model_dir:
            raise ValueError("Either --model or --model-dir must be specified")
        
        # Initialize evaluation pipeline
        config = {}
        if args.business_analysis:
            config['evaluation'] = {'business_analysis': True}
        if args.production_metrics:
            config['evaluation'] = config.get('evaluation', {})
            config['evaluation']['production_readiness'] = True
        if args.no_plots:
            config['output'] = {'save_plots': False}
        if args.no_reports:
            config['output'] = config.get('output', {})
            config['output']['save_reports'] = False
        
        pipeline = ModelEvaluationPipeline(config=config, logger=logger)
        
        # Determine models to evaluate
        model_paths = []
        if args.model:
            model_paths = [args.model]
        elif args.model_dir:
            model_paths = discover_models(args.model_dir)
            if not model_paths:
                raise ValueError(f"No models found in directory: {args.model_dir}")
        
        print(f"\n📋 Models to evaluate: {len(model_paths)}")
        for i, path in enumerate(model_paths, 1):
            print(f"  {i}. {Path(path).name}")
        
        # Load test data if provided
        X_test, y_test = None, None
        if args.test_data:
            print(f"\n📊 Loading test data from: {args.test_data}")
            X_test, y_test = pipeline.load_test_data(args.test_data, args.target_column)
        
        # Evaluate models
        print("\n🔍 Starting Model Evaluation...")
        model_results = {}
        
        for i, model_path in enumerate(model_paths, 1):
            print(f"\n  [{i}/{len(model_paths)}] Evaluating: {Path(model_path).name}")
            
            # Load model and metadata
            model, metadata = pipeline.load_model_with_metadata(model_path)
            model_name = metadata.get('model_name', Path(model_path).name)
            
            # Use test data from metadata if not provided
            if X_test is None:
                # Try to use model's saved test data or generate from training data
                logger.warning("No test data provided - using dummy evaluation")
                # For demo purposes, create dummy data
                np.random.seed(42)
                X_test = pd.DataFrame(np.random.randn(100, 10))
                y_test = pd.Series(np.random.choice([0, 1], 100))
            
            # Evaluate model
            result = pipeline.evaluate_single_model(model, X_test, y_test, model_name, metadata)
            model_results[model_name] = result
            
            print(f"    ✓ F1-Score: {result.f1_score:.3f}")
            
            if hasattr(result, 'production_ready'):
                ready = "✓" if result.production_ready['ready_for_production'] else "✗"
                print(f"    {ready} Production Ready")
        
        # Model comparison
        comparison_analysis = None
        if len(model_results) > 1:
            print("\n📈 Performing Model Comparison...")
            comparison_analysis = pipeline.compare_models(model_results)
            best_model = comparison_analysis['best_models']['best_overall']
            print(f"    🏆 Best Overall Model: {best_model}")
        
        # Generate visualizations
        print("\n📊 Generating Visualizations...")
        pipeline.generate_visualizations(model_results)
        
        # Save results and generate reports
        print("\n📋 Generating Evaluation Reports...")
        pipeline.save_evaluation_results(model_results, comparison_analysis)
        
        # Print summary
        evaluation_duration = time.time() - evaluation_start_time
        
        print(f"\n✅ Model Evaluation Completed Successfully!")
        print(f"⏱️  Total Time: {evaluation_duration:.2f} seconds")
        print(f"📁 Reports saved to: {args.output_dir}")
        
        if comparison_analysis:
            best_model = comparison_analysis['best_models']['best_overall']
            best_f1 = model_results[best_model].f1_score
            production_ready = comparison_analysis['summary_stats']['production_ready_count']
            
            print(f"🏆 Best Model: {best_model} (F1: {best_f1:.3f})")
            print(f"✅ Production Ready: {production_ready}/{len(model_results)} models")
        
        print("=" * 65)
        
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")
        print(f"\n❌ Evaluation pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
