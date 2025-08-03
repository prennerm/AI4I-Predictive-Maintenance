"""
Model Evaluation Framework Module

This module provides a comprehensive high-level evaluation framework that:
- Manages multiple models and their evaluation
- Generates detailed evaluation reports with visualizations
- Calculates business impact metrics
- Performs statistical model comparisons
- Creates production-ready evaluation pipelines

This is the high-level framework that uses model_utils.py utilities internally.

Author: AI4I Project Team
Created: August 2025
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import our utilities
from src.models.model_utils import (
    ModelEvaluator as BaseEvaluator, ModelPersistence, 
    FeatureImportanceAnalyzer, validate_model_inputs
)
from src.models.base_model import BaseModel, ClassificationModel, RegressionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation framework."""
    
    # Report settings
    output_dir: str = "reports"
    report_format: str = "html"  # 'html', 'pdf', 'both'
    include_plots: bool = True
    include_business_metrics: bool = True
    
    # Statistical analysis
    confidence_level: float = 0.95
    statistical_tests: bool = True
    
    # Business metrics
    failure_cost: float = 10000.0  # Cost of unplanned failure
    maintenance_cost: float = 1000.0  # Cost of planned maintenance
    false_positive_cost: float = 500.0  # Cost of unnecessary maintenance
    false_negative_cost: float = 8000.0  # Cost of missed failure
    
    # Visualization settings
    plot_style: str = "whitegrid"
    color_palette: str = "husl"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    
    # Performance thresholds
    min_accuracy: float = 0.85
    min_precision: float = 0.80
    min_recall: float = 0.85
    min_f1: float = 0.82


@dataclass
class ModelEvaluationResult:
    """Comprehensive evaluation result for a single model."""
    
    model_name: str
    model_type: str
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    
    # Detailed metrics
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    cross_val_scores: List[float]
    
    # Business metrics
    business_value: float
    cost_savings: float
    roi: float
    
    # Model characteristics
    training_time: float
    prediction_time: float
    model_size: int
    feature_importance: Dict[str, float]
    
    # Statistical significance
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Metadata
    evaluation_timestamp: str
    data_characteristics: Dict[str, Any]


class ModelEvaluator:
    """
    High-level model evaluation framework that orchestrates comprehensive evaluation.
    """
    
    def __init__(self, config: EvaluationConfig = None):
        """
        Initialize the evaluation framework.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        
        # Initialize components
        self.base_evaluator = BaseEvaluator()
        self.model_persistence = ModelPersistence()
        self.feature_analyzer = FeatureImportanceAnalyzer()
        
        # Storage for evaluation results
        self.models: Dict[str, BaseModel] = {}
        self.evaluation_results: Dict[str, ModelEvaluationResult] = {}
        self.comparison_data: Dict[str, Any] = {}
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup plotting
        plt.style.use(self.config.plot_style)
        sns.set_palette(self.config.color_palette)
        
        self.evaluation_id = self._generate_evaluation_id()
        logger.info(f"ModelEvaluator initialized with ID: {self.evaluation_id}")
    
    def _generate_evaluation_id(self) -> str:
        """Generate unique evaluation identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"evaluation_{timestamp}"
    
    def add_model(self, model: BaseModel, model_name: str = None) -> None:
        """
        Add a model to the evaluation framework.
        
        Args:
            model: Trained model instance
            model_name: Name for the model (defaults to class name)
        """
        if model_name is None:
            model_name = model.__class__.__name__
        
        self.models[model_name] = model
        logger.info(f"Added model: {model_name}")
    
    def load_models_from_directory(self, models_dir: str) -> None:
        """
        Load all models from a directory.
        
        Args:
            models_dir: Directory containing saved models
        """
        models_dir = Path(models_dir)
        
        if not models_dir.exists():
            logger.warning(f"Models directory does not exist: {models_dir}")
            return
        
        saved_models = self.model_persistence.list_saved_models()
        
        for model_info in saved_models:
            try:
                model_path = model_info['model_path']
                model, metadata = self.model_persistence.load_model(model_path)
                
                # Wrap in BaseModel for consistency
                wrapped_model = BaseModel()
                wrapped_model.model = model
                
                model_name = metadata.get('model_name', Path(model_path).name)
                self.add_model(wrapped_model, model_name)
                
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {str(e)}")
        
        logger.info(f"Loaded {len(self.models)} models from directory")
    
    def evaluate_model(self, 
                      model_name: str,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      X_train: np.ndarray = None,
                      y_train: np.ndarray = None,
                      feature_names: List[str] = None) -> ModelEvaluationResult:
        """
        Comprehensively evaluate a single model.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test feature matrix
            y_test: Test target vector
            X_train: Training features (optional, for cross-validation)
            y_train: Training targets (optional, for cross-validation)
            feature_names: Names of features
            
        Returns:
            Comprehensive evaluation result
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Add it first using add_model()")
        
        model = self.models[model_name]
        logger.info(f"Evaluating model: {model_name}")
        
        # Validate inputs
        validate_model_inputs(X_test, y_test)
        
        # Start timing
        start_time = datetime.now()
        
        # Make predictions
        pred_start = datetime.now()
        y_pred = model.predict(X_test)
        prediction_time = (datetime.now() - pred_start).total_seconds()
        
        # Get probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except:
                logger.warning(f"Could not get probabilities for {model_name}")
        
        # Basic evaluation using utilities
        if isinstance(model, ClassificationModel) or len(np.unique(y_test)) < 10:
            metrics = self.base_evaluator.evaluate_classification(y_test, y_pred, y_proba)
            problem_type = "classification"
        else:
            metrics = self.base_evaluator.evaluate_regression(y_test, y_pred)
            problem_type = "regression"
        
        # Cross-validation if training data provided
        cv_scores = []
        if X_train is not None and y_train is not None:
            try:
                cv_scores = cross_val_score(
                    model.model, X_train, y_train, cv=5, scoring='f1_weighted'
                ).tolist()
            except Exception as e:
                logger.warning(f"Cross-validation failed for {model_name}: {str(e)}")
        
        # Feature importance analysis
        feature_importance = {}
        if feature_names:
            feature_importance = self.feature_analyzer.get_feature_importance(
                model.model, feature_names
            )
        
        # Business metrics calculation
        business_metrics = self._calculate_business_metrics(y_test, y_pred, y_proba)
        
        # Statistical analysis
        confidence_intervals = self._calculate_confidence_intervals(
            y_test, y_pred, metrics
        )
        
        # Model characteristics
        model_size = self._get_model_size(model)
        training_time = getattr(model, 'training_time', 0.0)
        
        # Create result object
        if problem_type == "classification":
            result = ModelEvaluationResult(
                model_name=model_name,
                model_type=type(model.model).__name__,
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                roc_auc=metrics.get('roc_auc', 0.0),
                confusion_matrix=np.array(metrics['confusion_matrix']),
                classification_report=metrics['classification_report'],
                cross_val_scores=cv_scores,
                business_value=business_metrics['total_value'],
                cost_savings=business_metrics['cost_savings'],
                roi=business_metrics['roi'],
                training_time=training_time,
                prediction_time=prediction_time,
                model_size=model_size,
                feature_importance=feature_importance,
                confidence_intervals=confidence_intervals,
                evaluation_timestamp=start_time.isoformat(),
                data_characteristics={
                    'n_samples': len(X_test),
                    'n_features': X_test.shape[1],
                    'class_balance': dict(zip(*np.unique(y_test, return_counts=True)))
                }
            )
        else:
            # Regression metrics adaptation
            result = ModelEvaluationResult(
                model_name=model_name,
                model_type=type(model.model).__name__,
                accuracy=metrics['r2'],  # Use R² as accuracy for regression
                precision=1.0 - metrics['mape']/100,  # Inverse of MAPE
                recall=metrics['r2'],  # R² as recall proxy
                f1_score=metrics['r2'],  # R² as F1 proxy
                roc_auc=0.0,  # Not applicable for regression
                confusion_matrix=np.array([]),
                classification_report={},
                cross_val_scores=cv_scores,
                business_value=business_metrics['total_value'],
                cost_savings=business_metrics['cost_savings'],
                roi=business_metrics['roi'],
                training_time=training_time,
                prediction_time=prediction_time,
                model_size=model_size,
                feature_importance=feature_importance,
                confidence_intervals=confidence_intervals,
                evaluation_timestamp=start_time.isoformat(),
                data_characteristics={
                    'n_samples': len(X_test),
                    'n_features': X_test.shape[1],
                    'target_range': (float(np.min(y_test)), float(np.max(y_test)))
                }
            )
        
        # Store result
        self.evaluation_results[model_name] = result
        
        logger.info(f"Evaluation completed for {model_name}")
        logger.info(f"  - Accuracy: {result.accuracy:.4f}")
        logger.info(f"  - F1 Score: {result.f1_score:.4f}")
        logger.info(f"  - Business Value: ${result.business_value:,.2f}")
        
        return result
    
    def evaluate_all_models(self,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           X_train: np.ndarray = None,
                           y_train: np.ndarray = None,
                           feature_names: List[str] = None) -> Dict[str, ModelEvaluationResult]:
        """
        Evaluate all registered models.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
            X_train: Training features (optional)
            y_train: Training targets (optional)
            feature_names: Names of features
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating {len(self.models)} models...")
        
        results = {}
        for model_name in self.models:
            try:
                result = self.evaluate_model(
                    model_name, X_test, y_test, X_train, y_train, feature_names
                )
                results[model_name] = result
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        # Perform statistical comparisons
        self._perform_statistical_comparisons()
        
        logger.info(f"Evaluation completed for {len(results)} models")
        return results
    
    def _calculate_business_metrics(self, 
                                  y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  y_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate business impact metrics."""
        
        # Confusion matrix elements
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            
            # Business costs
            total_failures = len(y_true[y_true == 1])
            total_non_failures = len(y_true[y_true == 0])
            
            # Cost calculations
            prevented_failures_value = tp * self.config.failure_cost
            unnecessary_maintenance_cost = fp * self.config.false_positive_cost
            missed_failures_cost = fn * self.config.false_negative_cost
            maintenance_cost = (tp + fp) * self.config.maintenance_cost
            
            total_cost = unnecessary_maintenance_cost + missed_failures_cost + maintenance_cost
            total_benefit = prevented_failures_value
            net_value = total_benefit - total_cost
            
            # ROI calculation
            if total_cost > 0:
                roi = (net_value / total_cost) * 100
            else:
                roi = float('inf')
            
            # Cost savings compared to no prediction (all failures occur)
            baseline_cost = total_failures * self.config.failure_cost
            cost_savings = baseline_cost - total_cost
            
        else:
            # Multi-class or regression - simplified business metrics
            net_value = 0.0
            roi = 0.0
            cost_savings = 0.0
        
        return {
            'total_value': net_value,
            'roi': roi,
            'cost_savings': cost_savings
        }
    
    def _calculate_confidence_intervals(self, 
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      metrics: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key metrics."""
        
        n = len(y_true)
        confidence_intervals = {}
        
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        alpha = 1 - self.config.confidence_level
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric_name in metrics:
                bootstrap_scores = []
                
                for _ in range(n_bootstrap):
                    # Bootstrap sample
                    indices = np.random.choice(n, n, replace=True)
                    y_true_boot = y_true[indices]
                    y_pred_boot = y_pred[indices]
                    
                    # Calculate metric for bootstrap sample
                    if len(np.unique(y_true_boot)) > 1:  # Ensure both classes present
                        boot_metrics = self.base_evaluator.evaluate_classification(
                            y_true_boot, y_pred_boot
                        )
                        bootstrap_scores.append(boot_metrics[metric_name])
                
                if bootstrap_scores:
                    lower = np.percentile(bootstrap_scores, (alpha/2) * 100)
                    upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
                    confidence_intervals[metric_name] = (lower, upper)
        
        return confidence_intervals
    
    def _get_model_size(self, model: BaseModel) -> int:
        """Estimate model size in bytes."""
        try:
            import pickle
            return len(pickle.dumps(model.model))
        except:
            return 0
    
    def _perform_statistical_comparisons(self) -> None:
        """Perform statistical tests to compare model performance."""
        
        if len(self.evaluation_results) < 2:
            return
        
        logger.info("Performing statistical model comparisons...")
        
        # Collect cross-validation scores
        cv_data = {}
        for model_name, result in self.evaluation_results.items():
            if result.cross_val_scores:
                cv_data[model_name] = result.cross_val_scores
        
        # Pairwise statistical tests
        comparisons = {}
        model_names = list(cv_data.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                if model1 in cv_data and model2 in cv_data:
                    # Paired t-test
                    statistic, p_value = stats.ttest_rel(
                        cv_data[model1], cv_data[model2]
                    )
                    
                    comparisons[f"{model1}_vs_{model2}"] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < (1 - self.config.confidence_level),
                        'better_model': model1 if statistic > 0 else model2
                    }
        
        self.comparison_data['statistical_tests'] = comparisons
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate a comprehensive model comparison table."""
        
        if not self.evaluation_results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, result in self.evaluation_results.items():
            row = {
                'Model': model_name,
                'Model Type': result.model_type,
                'Accuracy': result.accuracy,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1 Score': result.f1_score,
                'ROC AUC': result.roc_auc,
                'Business Value ($)': result.business_value,
                'Cost Savings ($)': result.cost_savings,
                'ROI (%)': result.roi,
                'Training Time (s)': result.training_time,
                'Prediction Time (s)': result.prediction_time,
                'Model Size (KB)': result.model_size / 1024,
                'CV Mean': np.mean(result.cross_val_scores) if result.cross_val_scores else np.nan,
                'CV Std': np.std(result.cross_val_scores) if result.cross_val_scores else np.nan
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by F1 score (or business value)
        df = df.sort_values('F1 Score', ascending=False)
        
        return df
    
    def create_performance_visualizations(self) -> Dict[str, plt.Figure]:
        """Create comprehensive performance visualizations."""
        
        figures = {}
        
        if not self.evaluation_results:
            logger.warning("No evaluation results available for visualization")
            return figures
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = list(self.evaluation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            values = [getattr(self.evaluation_results[model], metric) for model in models]
            
            bars = ax.bar(models, values, color=sns.color_palette("husl", len(models)))
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        figures['performance_comparison'] = fig
        
        # 2. ROC Curves (for classification)
        classification_results = {
            name: result for name, result in self.evaluation_results.items()
            if result.roc_auc > 0
        }
        
        if classification_results:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            for model_name, result in classification_results.items():
                # Note: This is a simplified ROC curve visualization
                # In practice, you'd need the actual y_true, y_proba to plot real ROC curves
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curves Comparison')
                ax.legend()
            
            figures['roc_curves'] = fig
        
        # 3. Business Impact Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        business_values = [result.business_value for result in self.evaluation_results.values()]
        cost_savings = [result.cost_savings for result in self.evaluation_results.values()]
        
        # Business value
        bars1 = ax1.bar(models, business_values, color='green', alpha=0.7)
        ax1.set_title('Business Value by Model')
        ax1.set_ylabel('Net Business Value ($)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Cost savings
        bars2 = ax2.bar(models, cost_savings, color='blue', alpha=0.7)
        ax2.set_title('Cost Savings by Model')
        ax2.set_ylabel('Cost Savings ($)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        figures['business_impact'] = fig
        
        # 4. Feature Importance (for the best model)
        best_model_name = max(self.evaluation_results.keys(), 
                             key=lambda x: self.evaluation_results[x].f1_score)
        best_result = self.evaluation_results[best_model_name]
        
        if best_result.feature_importance:
            fig = self.feature_analyzer.plot_feature_importance(
                best_result.feature_importance,
                title=f'Feature Importance - {best_model_name}'
            )
            figures['feature_importance'] = fig
        
        return figures
    
    def generate_html_report(self) -> str:
        """Generate comprehensive HTML evaluation report."""
        
        report_path = self.output_dir / f"evaluation_report_{self.evaluation_id}.html"
        
        # Generate comparison table
        comparison_df = self.generate_comparison_table()
        
        # Create visualizations
        figures = self.create_performance_visualizations()
        
        # Save figures
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        figure_paths = {}
        for fig_name, fig in figures.items():
            fig_path = figures_dir / f"{fig_name}_{self.evaluation_id}.png"
            fig.savefig(fig_path, dpi=self.config.dpi, bbox_inches='tight')
            figure_paths[fig_name] = fig_path.relative_to(self.output_dir)
            plt.close(fig)
        
        # Generate HTML content
        html_content = self._generate_html_content(comparison_df, figure_paths)
        
        # Write HTML file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {report_path}")
        return str(report_path)
    
    def _generate_html_content(self, comparison_df: pd.DataFrame, figure_paths: Dict[str, Path]) -> str:
        """Generate HTML content for the evaluation report."""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report - {self.evaluation_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                          background-color: #e6f3ff; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; }}
                .warning {{ background-color: #fff3cd; }}
                .danger {{ background-color: #f8d7da; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Evaluation Report</h1>
                <p><strong>Evaluation ID:</strong> {self.evaluation_id}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Models Evaluated:</strong> {len(self.evaluation_results)}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                {self._generate_executive_summary()}
            </div>
            
            <div class="section">
                <h2>Model Comparison</h2>
                {comparison_df.to_html(classes='table table-striped', escape=False)}
            </div>
            
            <div class="section">
                <h2>Performance Visualizations</h2>
                {self._generate_figures_html(figure_paths)}
            </div>
            
            <div class="section">
                <h2>Detailed Results</h2>
                {self._generate_detailed_results_html()}
            </div>
            
            <div class="section">
                <h2>Business Impact Analysis</h2>
                {self._generate_business_analysis_html()}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {self._generate_recommendations_html()}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        
        if not self.evaluation_results:
            return "<p>No models were evaluated.</p>"
        
        # Find best model
        best_model = max(self.evaluation_results.items(), 
                        key=lambda x: x[1].f1_score)
        best_name, best_result = best_model
        
        # Calculate summary statistics
        avg_accuracy = np.mean([r.accuracy for r in self.evaluation_results.values()])
        total_business_value = sum([r.business_value for r in self.evaluation_results.values()])
        
        summary = f"""
        <div class="metric">
            <strong>Best Performing Model:</strong> {best_name}<br>
            <strong>F1 Score:</strong> {best_result.f1_score:.3f}<br>
            <strong>Business Value:</strong> ${best_result.business_value:,.2f}
        </div>
        <div class="metric">
            <strong>Average Accuracy:</strong> {avg_accuracy:.3f}<br>
            <strong>Total Business Value:</strong> ${total_business_value:,.2f}<br>
            <strong>Models Meeting Threshold:</strong> {self._count_models_meeting_thresholds()}
        </div>
        <p>The evaluation assessed {len(self.evaluation_results)} models across multiple performance 
        and business metrics. The best performing model is <strong>{best_name}</strong> with an 
        F1 score of {best_result.f1_score:.3f} and estimated business value of 
        ${best_result.business_value:,.2f}.</p>
        """
        
        return summary
    
    def _count_models_meeting_thresholds(self) -> str:
        """Count models meeting performance thresholds."""
        
        meets_all = 0
        for result in self.evaluation_results.values():
            if (result.accuracy >= self.config.min_accuracy and
                result.precision >= self.config.min_precision and
                result.recall >= self.config.min_recall and
                result.f1_score >= self.config.min_f1):
                meets_all += 1
        
        return f"{meets_all}/{len(self.evaluation_results)}"
    
    def _generate_figures_html(self, figure_paths: Dict[str, Path]) -> str:
        """Generate HTML for figures."""
        
        html = ""
        for fig_name, fig_path in figure_paths.items():
            title = fig_name.replace('_', ' ').title()
            html += f'<h3>{title}</h3>\n<img src="{fig_path}" alt="{title}">\n'
        
        return html
    
    def _generate_detailed_results_html(self) -> str:
        """Generate detailed results section."""
        
        html = ""
        for model_name, result in self.evaluation_results.items():
            html += f"""
            <h3>{model_name} ({result.model_type})</h3>
            <div class="metric">
                <strong>Performance Metrics:</strong><br>
                Accuracy: {result.accuracy:.3f}<br>
                Precision: {result.precision:.3f}<br>
                Recall: {result.recall:.3f}<br>
                F1 Score: {result.f1_score:.3f}
            </div>
            <div class="metric">
                <strong>Business Metrics:</strong><br>
                Business Value: ${result.business_value:,.2f}<br>
                Cost Savings: ${result.cost_savings:,.2f}<br>
                ROI: {result.roi:.1f}%
            </div>
            <div class="metric">
                <strong>Model Characteristics:</strong><br>
                Training Time: {result.training_time:.2f}s<br>
                Prediction Time: {result.prediction_time:.4f}s<br>
                Model Size: {result.model_size/1024:.1f} KB
            </div>
            """
        
        return html
    
    def _generate_business_analysis_html(self) -> str:
        """Generate business analysis section."""
        
        # Calculate aggregate business metrics
        total_value = sum([r.business_value for r in self.evaluation_results.values()])
        total_savings = sum([r.cost_savings for r in self.evaluation_results.values()])
        avg_roi = np.mean([r.roi for r in self.evaluation_results.values()])
        
        html = f"""
        <p>The predictive maintenance models demonstrate significant business value:</p>
        <ul>
            <li><strong>Total Business Value:</strong> ${total_value:,.2f}</li>
            <li><strong>Total Cost Savings:</strong> ${total_savings:,.2f}</li>
            <li><strong>Average ROI:</strong> {avg_roi:.1f}%</li>
        </ul>
        <p>These metrics are calculated based on the following assumptions:</p>
        <ul>
            <li>Failure Cost: ${self.config.failure_cost:,.2f}</li>
            <li>Maintenance Cost: ${self.config.maintenance_cost:,.2f}</li>
            <li>False Positive Cost: ${self.config.false_positive_cost:,.2f}</li>
            <li>False Negative Cost: ${self.config.false_negative_cost:,.2f}</li>
        </ul>
        """
        
        return html
    
    def _generate_recommendations_html(self) -> str:
        """Generate recommendations section."""
        
        # Find best model
        best_model = max(self.evaluation_results.items(), 
                        key=lambda x: x[1].f1_score)
        best_name, best_result = best_model
        
        html = f"""
        <h3>Model Selection Recommendation</h3>
        <p>Based on the comprehensive evaluation, <strong>{best_name}</strong> is recommended 
        for production deployment with the following justification:</p>
        <ul>
            <li>Highest F1 Score: {best_result.f1_score:.3f}</li>
            <li>Strong Business Value: ${best_result.business_value:,.2f}</li>
            <li>Acceptable Performance Characteristics</li>
        </ul>
        
        <h3>Implementation Considerations</h3>
        <ul>
            <li>Monitor model performance in production</li>
            <li>Retrain periodically with new data</li>
            <li>Validate business assumptions regularly</li>
            <li>Consider ensemble methods for improved robustness</li>
        </ul>
        """
        
        return html
    
    def save_evaluation_results(self, filename: str = None) -> str:
        """Save evaluation results to JSON file."""
        
        if filename is None:
            filename = f"evaluation_results_{self.evaluation_id}.json"
        
        output_path = self.output_dir / filename
        
        # Convert results to serializable format
        serializable_results = {}
        for model_name, result in self.evaluation_results.items():
            serializable_results[model_name] = {
                'model_name': result.model_name,
                'model_type': result.model_type,
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'roc_auc': result.roc_auc,
                'confusion_matrix': result.confusion_matrix.tolist(),
                'cross_val_scores': result.cross_val_scores,
                'business_value': result.business_value,
                'cost_savings': result.cost_savings,
                'roi': result.roi,
                'training_time': result.training_time,
                'prediction_time': result.prediction_time,
                'model_size': result.model_size,
                'feature_importance': result.feature_importance,
                'confidence_intervals': result.confidence_intervals,
                'evaluation_timestamp': result.evaluation_timestamp,
                'data_characteristics': result.data_characteristics
            }
        
        # Add comparison data
        output_data = {
            'evaluation_id': self.evaluation_id,
            'config': asdict(self.config),
            'results': serializable_results,
            'comparisons': self.comparison_data,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to: {output_path}")
        return str(output_path)
    
    def run_comprehensive_evaluation(self,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   X_train: np.ndarray = None,
                                   y_train: np.ndarray = None,
                                   feature_names: List[str] = None) -> str:
        """
        Run complete evaluation pipeline and generate all outputs.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
            X_train: Training features (optional)
            y_train: Training targets (optional)
            feature_names: Names of features
            
        Returns:
            Path to generated HTML report
        """
        logger.info("Running comprehensive evaluation pipeline...")
        
        # Evaluate all models
        self.evaluate_all_models(X_test, y_test, X_train, y_train, feature_names)
        
        # Generate outputs
        report_path = self.generate_html_report()
        results_path = self.save_evaluation_results()
        
        # Save comparison table
        comparison_df = self.generate_comparison_table()
        comparison_path = self.output_dir / f"model_comparison_{self.evaluation_id}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        logger.info("Comprehensive evaluation completed!")
        logger.info(f"  - HTML Report: {report_path}")
        logger.info(f"  - Results JSON: {results_path}")
        logger.info(f"  - Comparison CSV: {comparison_path}")
        
        return report_path


# Utility functions for standalone evaluation

def quick_evaluate_models(models: Dict[str, BaseModel],
                         X_test: np.ndarray,
                         y_test: np.ndarray,
                         output_dir: str = "reports") -> str:
    """
    Quick evaluation of multiple models with default settings.
    
    Args:
        models: Dictionary of model_name -> model
        X_test: Test features
        y_test: Test targets
        output_dir: Output directory
        
    Returns:
        Path to generated report
    """
    evaluator = ModelEvaluator(EvaluationConfig(output_dir=output_dir))
    
    for name, model in models.items():
        evaluator.add_model(model, name)
    
    return evaluator.run_comprehensive_evaluation(X_test, y_test)


def load_and_evaluate_models(models_dir: str,
                           test_data_path: str,
                           target_column: str = 'target',
                           output_dir: str = "reports") -> str:
    """
    Load models from directory and evaluate on test data.
    
    Args:
        models_dir: Directory containing saved models
        test_data_path: Path to test data CSV
        target_column: Name of target column
        output_dir: Output directory
        
    Returns:
        Path to generated report
    """
    # Load test data
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop(columns=[target_column]).values
    y_test = test_data[target_column].values
    feature_names = test_data.drop(columns=[target_column]).columns.tolist()
    
    # Initialize evaluator and load models
    evaluator = ModelEvaluator(EvaluationConfig(output_dir=output_dir))
    evaluator.load_models_from_directory(models_dir)
    
    return evaluator.run_comprehensive_evaluation(X_test, y_test, feature_names=feature_names)


# Testing function
def test_model_evaluator():
    """Test function for ModelEvaluator."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    
    print("Testing ModelEvaluator...")
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    svm_model = SVC(probability=True, random_state=42)
    
    rf_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    
    # Wrap models
    rf_wrapped = BaseModel()
    rf_wrapped.model = rf_model
    
    svm_wrapped = BaseModel()
    svm_wrapped.model = svm_model
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    evaluator.add_model(rf_wrapped, "RandomForest")
    evaluator.add_model(svm_wrapped, "SVM")
    
    # Run evaluation
    report_path = evaluator.run_comprehensive_evaluation(X_test, y_test, X_train, y_train)
    
    print(f"Evaluation completed! Report saved to: {report_path}")
    
    # Check results
    assert len(evaluator.evaluation_results) == 2
    assert Path(report_path).exists()
    
    print("All tests passed!")


if __name__ == "__main__":
    test_model_evaluator()
