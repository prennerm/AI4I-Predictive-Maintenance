"""
Model Plots Module

This module provides comprehensive model performance visualizations
specifically optimized for predictive maintenance machine learning models:

- Classification performance metrics (ROC, PR curves, confusion matrices)
- Regression performance analysis (residuals, prediction plots)
- Model comparison visualizations
- Feature importance and interpretability plots
- Learning curves and validation analysis
- Business-oriented performance metrics
- Interactive model exploration dashboards

All plots are designed to support both technical analysis and business
decision-making for predictive maintenance applications.

Author: AI4I Project Team
Created: August 2025
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings

# Scikit-learn imports
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.calibration import calibration_curve

# Advanced plotting
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Install with: pip install plotly")

# Feature importance (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPlotter:
    """
    Comprehensive model performance visualization class for AI4I Predictive Maintenance.
    
    This class provides all essential model evaluation visualizations with both static 
    and interactive options, specifically tailored for predictive maintenance analysis.
    """
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (12, 8),
                 style: str = 'seaborn-v0_8',
                 color_palette: str = 'husl',
                 save_plots: bool = True,
                 output_dir: str = 'reports/figures',
                 dpi: int = 300,
                 interactive: bool = False):
        """
        Initialize Model Plotter.
        
        Args:
            figsize: Default figure size for matplotlib plots
            style: Matplotlib style
            color_palette: Seaborn color palette
            save_plots: Whether to automatically save plots
            output_dir: Directory to save plots
            dpi: Resolution for saved plots
            interactive: Use interactive plotly plots when available
        """
        self.figsize = figsize
        self.style = style
        self.color_palette = color_palette
        self.save_plots = save_plots
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.interactive = interactive and PLOTLY_AVAILABLE
        
        # Create output directory
        if self.save_plots:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plotting style
        plt.style.use(self.style)
        sns.set_palette(self.color_palette)
        
        # Color schemes for different model performance aspects
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#4ECDC4',
            'warning': '#F18F01',
            'danger': '#FF6B6B',
            'info': '#74B9FF',
            'neutral': '#6C757D',
            'excellent': '#00B894',
            'good': '#FDCB6E',
            'poor': '#E17055'
        }
        
        # Performance thresholds for color coding
        self.thresholds = {
            'accuracy': {'excellent': 0.95, 'good': 0.85, 'acceptable': 0.75},
            'precision': {'excellent': 0.9, 'good': 0.8, 'acceptable': 0.7},
            'recall': {'excellent': 0.9, 'good': 0.8, 'acceptable': 0.7},
            'f1': {'excellent': 0.9, 'good': 0.8, 'acceptable': 0.7},
            'auc': {'excellent': 0.95, 'good': 0.85, 'acceptable': 0.75}
        }
        
        logger.info(f"ModelPlotter initialized with {'interactive' if self.interactive else 'static'} plotting")
    
    def _get_performance_color(self, metric: str, value: float) -> str:
        """Get color based on performance threshold."""
        if metric in self.thresholds:
            thresholds = self.thresholds[metric]
            if value >= thresholds['excellent']:
                return self.colors['excellent']
            elif value >= thresholds['good']:
                return self.colors['good']
            elif value >= thresholds['acceptable']:
                return self.colors['warning']
            else:
                return self.colors['poor']
        return self.colors['primary']
    
    def plot_classification_performance(self, 
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray, 
                                      y_proba: Optional[np.ndarray] = None,
                                      model_name: str = "Model",
                                      class_names: Optional[List[str]] = None) -> None:
        """
        Comprehensive classification performance visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (for ROC/PR curves)
            model_name: Name of the model
            class_names: Names of the classes
        """
        logger.info(f"Creating classification performance plots for {model_name}...")
        
        if class_names is None:
            class_names = [f"Class {i}" for i in np.unique(y_true)]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Classification Performance: {model_name}", fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=class_names, yticklabels=class_names)
        ax1.set_title("Confusion Matrix", fontweight='bold')
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        
        # Add accuracy on confusion matrix
        ax1.text(0.5, -0.1, f"Accuracy: {accuracy:.3f}", 
                transform=ax1.transAxes, ha='center', fontsize=12, fontweight='bold')
        
        # 2. Metrics Bar Chart
        ax2 = axes[0, 1]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1]
        colors = [self._get_performance_color(m.lower(), v) for m, v in zip(metrics, values)]
        
        bars = ax2.bar(metrics, values, color=colors, alpha=0.8)
        ax2.set_title("Performance Metrics", fontweight='bold')
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. ROC Curve (if probabilities available)
        ax3 = axes[0, 2]
        if y_proba is not None and len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            roc_auc = auc(fpr, tpr)
            
            ax3.plot(fpr, tpr, color=self.colors['primary'], linewidth=2, 
                    label=f'ROC Curve (AUC = {roc_auc:.3f})')
            ax3.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.8)
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curve', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "ROC Curve\nRequires binary classification\nwith probabilities", 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("ROC Curve", fontweight='bold')
        
        # 4. Precision-Recall Curve
        ax4 = axes[1, 0]
        if y_proba is not None and len(np.unique(y_true)) == 2:
            precision_vals, recall_vals, _ = precision_recall_curve(
                y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            avg_precision = average_precision_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            
            ax4.plot(recall_vals, precision_vals, color=self.colors['secondary'], linewidth=2,
                    label=f'PR Curve (AP = {avg_precision:.3f})')
            ax4.set_xlim([0.0, 1.0])
            ax4.set_ylim([0.0, 1.05])
            ax4.set_xlabel('Recall')
            ax4.set_ylabel('Precision')
            ax4.set_title('Precision-Recall Curve', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "PR Curve\nRequires binary classification\nwith probabilities", 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Precision-Recall Curve", fontweight='bold')
        
        # 5. Classification Report (as text)
        ax5 = axes[1, 1]
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_text = classification_report(y_true, y_pred, target_names=class_names)
        
        ax5.text(0.1, 0.9, report_text, fontsize=10, verticalalignment='top',
                transform=ax5.transAxes, fontfamily='monospace',
                bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.5))
        ax5.set_title("Detailed Classification Report", fontweight='bold')
        ax5.axis('off')
        
        # 6. Feature Importance (placeholder or prediction distribution)
        ax6 = axes[1, 2]
        if y_proba is not None:
            # Prediction probability distribution
            if y_proba.ndim > 1:
                prob_positive = y_proba[:, 1]
            else:
                prob_positive = y_proba
            
            # Separate by true class
            prob_positive_true = prob_positive[y_true == 1]
            prob_positive_false = prob_positive[y_true == 0]
            
            ax6.hist(prob_positive_false, bins=30, alpha=0.7, color=self.colors['success'], 
                    label=f'{class_names[0]} (n={len(prob_positive_false)})', density=True)
            ax6.hist(prob_positive_true, bins=30, alpha=0.7, color=self.colors['danger'], 
                    label=f'{class_names[1]} (n={len(prob_positive_true)})', density=True)
            ax6.set_xlabel('Prediction Probability')
            ax6.set_ylabel('Density')
            ax6.set_title('Prediction Probability Distribution', fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, "Prediction Distribution\nRequires probabilities", 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title("Prediction Distribution", fontweight='bold')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.output_dir / f'classification_performance_{model_name.replace(" ", "_")}.png', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_regression_performance(self, 
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  model_name: str = "Model") -> None:
        """
        Comprehensive regression performance visualization.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
        """
        logger.info(f"Creating regression performance plots for {model_name}...")
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Regression Performance: {model_name}", fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted
        ax1 = axes[0, 0]
        ax1.scatter(y_true, y_pred, alpha=0.6, color=self.colors['primary'])
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Actual vs Predicted', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add R² on plot
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                fontsize=12, fontweight='bold')
        
        # 2. Residuals vs Predicted
        ax2 = axes[0, 1]
        ax2.scatter(y_pred, residuals, alpha=0.6, color=self.colors['secondary'])
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Predicted', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals Distribution
        ax3 = axes[0, 2]
        ax3.hist(residuals, bins=30, alpha=0.7, color=self.colors['info'], edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residuals Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add mean and std of residuals
        ax3.text(0.05, 0.95, f'Mean: {residuals.mean():.3f}\nStd: {residuals.std():.3f}', 
                transform=ax3.transAxes, 
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='top')
        
        # 4. Q-Q Plot for residuals normality
        ax4 = axes[1, 0]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Residuals Normality)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance Metrics
        ax5 = axes[1, 1]
        metrics = ['R²', 'RMSE', 'MAE']
        values = [r2, rmse, mae]
        
        # Normalize RMSE and MAE for display (relative to target range)
        target_range = y_true.max() - y_true.min()
        display_values = [r2, rmse/target_range, mae/target_range]
        colors = [self.colors['excellent'] if r2 > 0.8 else self.colors['good'] if r2 > 0.6 else self.colors['warning'],
                 self.colors['primary'], self.colors['primary']]
        
        bars = ax5.bar(metrics, display_values, color=colors, alpha=0.8)
        ax5.set_title("Performance Metrics", fontweight='bold')
        ax5.set_ylabel("Score")
        ax5.grid(True, alpha=0.3)
        
        # Add actual values as text
        for i, (bar, actual_val) in enumerate(zip(bars, values)):
            if i == 0:  # R²
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{actual_val:.3f}', ha='center', va='bottom', fontweight='bold')
            else:  # RMSE, MAE
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{actual_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Error Analysis
        ax6 = axes[1, 2]
        
        # Absolute errors
        abs_errors = np.abs(residuals)
        
        # Categorize errors
        error_percentiles = np.percentile(abs_errors, [50, 75, 90, 95])
        
        error_stats = f"""
Error Analysis:

RMSE: {rmse:.3f}
MAE: {mae:.3f}
R²: {r2:.3f}

Error Percentiles:
50th: {error_percentiles[0]:.3f}
75th: {error_percentiles[1]:.3f}
90th: {error_percentiles[2]:.3f}
95th: {error_percentiles[3]:.3f}

Max Error: {abs_errors.max():.3f}
"""
        
        ax6.text(0.1, 0.9, error_stats, fontsize=11, verticalalignment='top',
                transform=ax6.transAxes, 
                bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.5))
        ax6.set_title("Error Statistics", fontweight='bold')
        ax6.axis('off')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.output_dir / f'regression_performance_{model_name.replace(" ", "_")}.png', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, 
                            results: Dict[str, Dict[str, float]],
                            metric_type: str = 'classification',
                            title: str = "Model Comparison") -> None:
        """
        Compare multiple models performance.
        
        Args:
            results: Dictionary with model names as keys and metrics as values
            metric_type: 'classification' or 'regression'
            title: Plot title
        """
        logger.info(f"Creating model comparison plots for {len(results)} models...")
        
        if not results:
            logger.warning("No results provided for model comparison")
            return
        
        # Extract model names and metrics
        model_names = list(results.keys())
        
        if metric_type == 'classification':
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
            metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        else:  # regression
            metrics = ['r2_score', 'rmse', 'mae', 'mape']
            metric_labels = ['R²', 'RMSE', 'MAE', 'MAPE']
        
        # Prepare data
        comparison_data = []
        for model_name in model_names:
            for metric in metrics:
                if metric in results[model_name]:
                    comparison_data.append({
                        'Model': model_name,
                        'Metric': metric,
                        'Value': results[model_name][metric]
                    })
        
        if not comparison_data:
            logger.warning("No valid metrics found for comparison")
            return
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Grouped Bar Chart
        ax1 = axes[0, 0]
        
        # Pivot data for grouped bar chart
        pivot_df = comparison_df.pivot(index='Model', columns='Metric', values='Value')
        
        # Plot grouped bars
        pivot_df.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title("Performance Metrics by Model", fontweight='bold')
        ax1.set_xlabel("Models")
        ax1.set_ylabel("Score")
        ax1.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Heatmap
        ax2 = axes[0, 1]
        sns.heatmap(pivot_df.T, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2)
        ax2.set_title("Performance Heatmap", fontweight='bold')
        ax2.set_xlabel("Models")
        ax2.set_ylabel("Metrics")
        
        # 3. Radar Chart (if multiple metrics)
        ax3 = axes[1, 0]
        if len(metrics) >= 3:
            self._plot_radar_chart(pivot_df, ax3)
        else:
            ax3.text(0.5, 0.5, "Radar Chart\nRequires 3+ metrics", 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("Model Performance Radar", fontweight='bold')
        
        # 4. Best Model Summary
        ax4 = axes[1, 1]
        
        # Find best model for each metric
        best_models = {}
        for metric in metrics:
            if metric in pivot_df.columns:
                if metric in ['rmse', 'mae', 'mape']:  # Lower is better
                    best_model = pivot_df[metric].idxmin()
                    best_value = pivot_df[metric].min()
                else:  # Higher is better
                    best_model = pivot_df[metric].idxmax()
                    best_value = pivot_df[metric].max()
                best_models[metric] = (best_model, best_value)
        
        summary_text = "Best Models by Metric:\n\n"
        for metric, (model, value) in best_models.items():
            summary_text += f"{metric.upper()}: {model}\n"
            summary_text += f"  Value: {value:.3f}\n\n"
        
        # Overall best model (most frequent winner)
        model_wins = {}
        for metric, (model, value) in best_models.items():
            model_wins[model] = model_wins.get(model, 0) + 1
        
        overall_best = max(model_wins, key=model_wins.get)
        summary_text += f"Overall Best: {overall_best}\n"
        summary_text += f"Wins: {model_wins[overall_best]}/{len(best_models)} metrics"
        
        ax4.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                transform=ax4.transAxes, 
                bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.5))
        ax4.set_title("Best Model Summary", fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.output_dir / f'model_comparison_{metric_type}.png', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def _plot_radar_chart(self, data: pd.DataFrame, ax) -> None:
        """Create radar chart for model comparison."""
        
        # Normalize data to 0-1 scale for radar chart
        normalized_data = data.copy()
        for col in data.columns:
            if col in ['rmse', 'mae', 'mape']:  # Lower is better - invert
                normalized_data[col] = 1 - (data[col] - data[col].min()) / (data[col].max() - data[col].min())
            else:  # Higher is better
                normalized_data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        
        # Number of metrics
        num_metrics = len(normalized_data.columns)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax.clear()
        
        # Plot each model
        colors = plt.cm.Set1(np.linspace(0, 1, len(normalized_data)))
        for i, (model_name, row) in enumerate(normalized_data.iterrows()):
            values = row.tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Add metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(normalized_data.columns)
        ax.set_ylim(0, 1)
        ax.set_title("Model Performance Radar", fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
    
    def plot_learning_curves(self, 
                           estimator,
                           X: np.ndarray,
                           y: np.ndarray,
                           cv: int = 5,
                           scoring: str = 'accuracy',
                           model_name: str = "Model") -> None:
        """
        Plot learning curves to analyze model performance vs training size.
        
        Args:
            estimator: Trained model
            X: Features
            y: Target
            cv: Cross-validation folds
            scoring: Scoring metric
            model_name: Name of the model
        """
        logger.info(f"Creating learning curves for {model_name}...")
        
        # Calculate learning curves
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=-1, 
            train_sizes=train_sizes, scoring=scoring
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Learning Curves: {model_name}", fontsize=16, fontweight='bold')
        
        # 1. Learning Curve
        ax1 = axes[0]
        ax1.plot(train_sizes_abs, train_mean, 'o-', color=self.colors['primary'], 
                label=f'Training {scoring}')
        ax1.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color=self.colors['primary'])
        
        ax1.plot(train_sizes_abs, val_mean, 'o-', color=self.colors['secondary'], 
                label=f'Validation {scoring}')
        ax1.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color=self.colors['secondary'])
        
        ax1.set_xlabel('Training Set Size')
        ax1.set_ylabel(f'{scoring.title()} Score')
        ax1.set_title('Learning Curve', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance Gap Analysis
        ax2 = axes[1]
        performance_gap = train_mean - val_mean
        ax2.plot(train_sizes_abs, performance_gap, 'o-', color=self.colors['warning'], 
                linewidth=2, label='Performance Gap')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Training Set Size')
        ax2.set_ylabel('Training - Validation Score')
        ax2.set_title('Overfitting Analysis', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add interpretation text
        final_gap = performance_gap[-1]
        if final_gap > 0.1:
            interpretation = "High overfitting"
            color = self.colors['danger']
        elif final_gap > 0.05:
            interpretation = "Moderate overfitting"
            color = self.colors['warning']
        else:
            interpretation = "Good generalization"
            color = self.colors['success']
        
        ax2.text(0.05, 0.95, f'Final Gap: {final_gap:.3f}\n{interpretation}', 
                transform=ax2.transAxes, 
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.3),
                fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.output_dir / f'learning_curves_{model_name.replace(" ", "_")}.png', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, 
                              feature_names: List[str],
                              importance_scores: np.ndarray,
                              model_name: str = "Model",
                              top_n: int = 20) -> None:
        """
        Plot feature importance analysis.
        
        Args:
            feature_names: Names of the features
            importance_scores: Importance scores
            model_name: Name of the model
            top_n: Number of top features to show
        """
        logger.info(f"Creating feature importance plots for {model_name}...")
        
        # Create feature importance dataframe
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        # Limit to top N features
        top_features = feature_df.head(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f"Feature Importance: {model_name}", fontsize=16, fontweight='bold')
        
        # 1. Horizontal bar chart
        ax1 = axes[0]
        bars = ax1.barh(range(len(top_features)), top_features['Importance'], 
                       color=self.colors['primary'])
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['Feature'])
        ax1.set_xlabel('Importance Score')
        ax1.set_title(f'Top {top_n} Features', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, top_features['Importance']):
            ax1.text(bar.get_width() + max(top_features['Importance']) * 0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left', va='center', fontsize=9)
        
        # 2. Cumulative importance
        ax2 = axes[1]
        cumulative_importance = np.cumsum(feature_df['Importance']) / np.sum(feature_df['Importance'])
        
        ax2.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
                'o-', color=self.colors['secondary'], linewidth=2)
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.8, label='80% threshold')
        ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.8, label='90% threshold')
        
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance')
        ax2.set_title('Cumulative Feature Importance', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Find features needed for 80% and 90% importance
        features_80 = np.argmax(cumulative_importance >= 0.8) + 1
        features_90 = np.argmax(cumulative_importance >= 0.9) + 1
        
        ax2.text(0.05, 0.95, f'Features for 80%: {features_80}\nFeatures for 90%: {features_90}', 
                transform=ax2.transAxes, 
                bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.5),
                fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.output_dir / f'feature_importance_{model_name.replace(" ", "_")}.png', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_calibration_curve(self, 
                             y_true: np.ndarray,
                             y_proba: np.ndarray,
                             model_name: str = "Model",
                             n_bins: int = 10) -> None:
        """
        Plot calibration curve for probability predictions.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            n_bins: Number of bins for calibration
        """
        logger.info(f"Creating calibration curve for {model_name}...")
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Probability Calibration: {model_name}", fontsize=16, fontweight='bold')
        
        # 1. Calibration curve
        ax1 = axes[0]
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", 
                color=self.colors['primary'], linewidth=2, label=model_name)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Curve', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Prediction histogram
        ax2 = axes[1]
        ax2.hist(y_proba, bins=20, alpha=0.7, color=self.colors['info'], 
                edgecolor='black', density=True)
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Density')
        ax2.set_title('Prediction Probability Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add calibration metrics
        from sklearn.metrics import brier_score_loss
        brier_score = brier_score_loss(y_true, y_proba)
        
        ax2.text(0.05, 0.95, f'Brier Score: {brier_score:.3f}\n(Lower is better)', 
                transform=ax2.transAxes, 
                bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.7),
                fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.output_dir / f'calibration_curve_{model_name.replace(" ", "_")}.png', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_business_metrics(self, 
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            cost_matrix: Optional[Dict[str, float]] = None,
                            model_name: str = "Model") -> None:
        """
        Plot business-oriented metrics for predictive maintenance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_matrix: Business costs {'tp': cost, 'tn': cost, 'fp': cost, 'fn': cost}
            model_name: Name of the model
        """
        logger.info(f"Creating business metrics visualization for {model_name}...")
        
        # Default cost matrix for predictive maintenance
        if cost_matrix is None:
            cost_matrix = {
                'tp': -100,  # Correct failure prediction saves money
                'tn': -10,   # Correct normal prediction (small operational cost)
                'fp': 500,   # False alarm costs unnecessary maintenance
                'fn': 5000   # Missing failure is very expensive
            }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate business metrics
        total_cost = (tp * cost_matrix['tp'] + 
                     tn * cost_matrix['tn'] + 
                     fp * cost_matrix['fp'] + 
                     fn * cost_matrix['fn'])
        
        # Calculate savings compared to reactive maintenance
        reactive_cost = len(y_true) * cost_matrix['fn']  # Assume all failures are missed
        savings = reactive_cost - total_cost
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Business Impact Analysis: {model_name}", fontsize=16, fontweight='bold')
        
        # 1. Cost Matrix Visualization
        ax1 = axes[0, 0]
        cost_cm = np.array([[tn * cost_matrix['tn'], fp * cost_matrix['fp']],
                           [fn * cost_matrix['fn'], tp * cost_matrix['tp']]])
        
        sns.heatmap(cost_cm, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=ax1,
                   xticklabels=['Predicted Normal', 'Predicted Failure'],
                   yticklabels=['Actual Normal', 'Actual Failure'])
        ax1.set_title('Business Cost Matrix', fontweight='bold')
        
        # 2. Cost Breakdown
        ax2 = axes[0, 1]
        costs = [tp * cost_matrix['tp'], tn * cost_matrix['tn'], 
                fp * cost_matrix['fp'], fn * cost_matrix['fn']]
        labels = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
        colors = [self.colors['excellent'], self.colors['success'], 
                 self.colors['warning'], self.colors['danger']]
        
        bars = ax2.bar(labels, costs, color=colors, alpha=0.8)
        ax2.set_title('Cost Breakdown by Prediction Type', fontweight='bold')
        ax2.set_ylabel('Total Cost')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, cost in zip(bars, costs):
            ax2.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (max(costs) - min(costs)) * 0.01,
                    f'${cost:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. ROI Analysis
        ax3 = axes[1, 0]
        
        roi_data = {
            'Scenario': ['Reactive\nMaintenance', 'Predictive\nMaintenance', 'Savings'],
            'Cost': [reactive_cost, total_cost, savings],
            'Color': [self.colors['danger'], self.colors['primary'], self.colors['excellent']]
        }
        
        bars = ax3.bar(roi_data['Scenario'], roi_data['Cost'], 
                      color=roi_data['Color'], alpha=0.8)
        ax3.set_title('ROI Analysis', fontweight='bold')
        ax3.set_ylabel('Total Cost')
        
        # Add value labels
        for bar, cost in zip(bars, roi_data['Cost']):
            if cost >= 0:
                va = 'bottom'
                y_pos = bar.get_height() + max(roi_data['Cost']) * 0.01
            else:
                va = 'top'
                y_pos = bar.get_height() - max(roi_data['Cost']) * 0.01
            
            ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'${cost:,.0f}', ha='center', va=va, fontweight='bold')
        
        # 4. Business Summary
        ax4 = axes[1, 1]
        
        # Calculate additional business metrics
        failure_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        cost_per_prediction = total_cost / len(y_true) if len(y_true) > 0 else 0
        
        summary_text = f"""
Business Impact Summary:

Total Cost: ${total_cost:,.0f}
Reactive Cost: ${reactive_cost:,.0f}
Net Savings: ${savings:,.0f}
ROI: {(savings/abs(reactive_cost)*100):.1f}%

Operational Metrics:
• Failure Detection Rate: {failure_detection_rate:.1%}
• False Alarm Rate: {false_alarm_rate:.1%}
• Cost per Prediction: ${cost_per_prediction:.2f}

Predictions Breakdown:
• True Positives: {tp:,} (Correct failures)
• False Negatives: {fn:,} (Missed failures)
• False Positives: {fp:,} (False alarms)
• True Negatives: {tn:,} (Correct normals)
"""
        
        ax4.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                transform=ax4.transAxes, 
                bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.5))
        ax4.set_title("Business Summary", fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.output_dir / f'business_metrics_{model_name.replace(" ", "_")}.png', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def generate_model_report(self, 
                            model_results: Dict[str, Any],
                            model_name: str = "Model",
                            problem_type: str = 'classification') -> None:
        """
        Generate comprehensive model performance report.
        
        Args:
            model_results: Dictionary containing all model results
            model_name: Name of the model
            problem_type: 'classification' or 'regression'
        """
        logger.info(f"Generating comprehensive model report for {model_name}...")
        
        print(f"\n{'='*80}")
        print(f"MODEL PERFORMANCE REPORT: {model_name}".center(80))
        print(f"{'='*80}\n")
        
        if problem_type == 'classification':
            # Classification performance
            if all(key in model_results for key in ['y_true', 'y_pred']):
                print("📊 CLASSIFICATION PERFORMANCE")
                print("-" * 50)
                self.plot_classification_performance(
                    model_results['y_true'], 
                    model_results['y_pred'],
                    model_results.get('y_proba'),
                    model_name
                )
            
            # Calibration analysis
            if 'y_proba' in model_results:
                print("\n🎯 PROBABILITY CALIBRATION")
                print("-" * 50)
                self.plot_calibration_curve(
                    model_results['y_true'],
                    model_results['y_proba'][:, 1] if model_results['y_proba'].ndim > 1 else model_results['y_proba'],
                    model_name
                )
            
            # Business metrics
            print("\n💰 BUSINESS IMPACT ANALYSIS")
            print("-" * 50)
            self.plot_business_metrics(
                model_results['y_true'],
                model_results['y_pred'],
                model_results.get('cost_matrix'),
                model_name
            )
        
        else:  # regression
            print("📈 REGRESSION PERFORMANCE")
            print("-" * 50)
            self.plot_regression_performance(
                model_results['y_true'],
                model_results['y_pred'],
                model_name
            )
        
        # Feature importance
        if 'feature_importance' in model_results and 'feature_names' in model_results:
            print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
            print("-" * 50)
            self.plot_feature_importance(
                model_results['feature_names'],
                model_results['feature_importance'],
                model_name
            )
        
        # Learning curves
        if all(key in model_results for key in ['estimator', 'X', 'y']):
            print("\n📚 LEARNING CURVE ANALYSIS")
            print("-" * 50)
            self.plot_learning_curves(
                model_results['estimator'],
                model_results['X'],
                model_results['y'],
                model_name=model_name
            )
        
        print(f"\n{'='*80}")
        print("MODEL REPORT COMPLETED")
        print(f"All plots saved to: {self.output_dir}")
        print(f"{'='*80}\n")


# Convenience functions
def plot_model_performance(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_proba: Optional[np.ndarray] = None,
                         model_name: str = "Model",
                         problem_type: str = 'classification',
                         save_plots: bool = True) -> None:
    """
    Quick model performance visualization.
    
    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        y_proba: Prediction probabilities (for classification)
        model_name: Name of the model
        problem_type: 'classification' or 'regression'
        save_plots: Whether to save plots
    """
    plotter = ModelPlotter(save_plots=save_plots)
    
    if problem_type == 'classification':
        plotter.plot_classification_performance(y_true, y_pred, y_proba, model_name)
    else:
        plotter.plot_regression_performance(y_true, y_pred, model_name)


def compare_models(results_dict: Dict[str, Dict[str, float]],
                  metric_type: str = 'classification',
                  save_plots: bool = True) -> None:
    """
    Quick model comparison visualization.
    
    Args:
        results_dict: Dictionary with model results
        metric_type: 'classification' or 'regression'
        save_plots: Whether to save plots
    """
    plotter = ModelPlotter(save_plots=save_plots)
    plotter.plot_model_comparison(results_dict, metric_type)


# Testing function
def test_model_plots():
    """Test model plotting functionality."""
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    print("Testing model plots...")
    
    # Test classification plots
    X_cls, y_cls = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    
    rf_cls = RandomForestClassifier(random_state=42)
    rf_cls.fit(X_train, y_train)
    y_pred_cls = rf_cls.predict(X_test)
    y_proba_cls = rf_cls.predict_proba(X_test)
    
    plotter = ModelPlotter(save_plots=False)
    plotter.plot_classification_performance(y_test, y_pred_cls, y_proba_cls, "Random Forest")
    
    # Test regression plots
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    rf_reg = RandomForestRegressor(random_state=42)
    rf_reg.fit(X_train, y_train)
    y_pred_reg = rf_reg.predict(X_test)
    
    plotter.plot_regression_performance(y_test, y_pred_reg, "Random Forest")
    
    print("Model plots testing completed successfully!")


if __name__ == "__main__":
    test_model_plots()
