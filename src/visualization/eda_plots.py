"""
EDA Plots Module

This module provides comprehensive exploratory data analysis visualizations
specifically optimized for the AI4I 2020 Predictive Maintenance dataset:

- Dataset overview and structure analysis
- Distribution analysis for all feature types
- Correlation analysis and heatmaps
- Target variable analysis and failure patterns
- Time-series analysis and trends
- Statistical summaries with visual components
- Interactive plots for detailed exploration

All plots are designed to provide actionable insights for predictive maintenance
modeling and feature engineering decisions.

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

# Scientific computing
from scipy import stats
from scipy.stats import normaltest, jarque_bera, shapiro

# Interactive plotting (optional)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Install with: pip install plotly")

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EDAPlotter:
    """
    Comprehensive EDA plotting class for AI4I Predictive Maintenance dataset.
    
    This class provides all essential EDA visualizations with both static (matplotlib/seaborn)
    and interactive (plotly) options, specifically tailored for predictive maintenance analysis.
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
        Initialize EDA Plotter.
        
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
        
        # Color schemes for different plot types
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'neutral': '#6C757D',
            'failure': '#FF6B6B',
            'normal': '#4ECDC4'
        }
        
        logger.info(f"EDAPlotter initialized with {'interactive' if self.interactive else 'static'} plotting")
    
    def plot_dataset_overview(self, df: pd.DataFrame, title: str = "AI4I Dataset Overview") -> None:
        """
        Create comprehensive dataset overview plots.
        
        Args:
            df: Input dataframe
            title: Plot title
        """
        logger.info("Creating dataset overview plots...")
        
        if self.interactive and PLOTLY_AVAILABLE:
            self._plot_dataset_overview_interactive(df, title)
        else:
            self._plot_dataset_overview_static(df, title)
    
    def _plot_dataset_overview_static(self, df: pd.DataFrame, title: str) -> None:
        """Static dataset overview plots."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Dataset shape and info
        ax1 = axes[0, 0]
        info_text = f"""
Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns

Data Types:
• Numeric: {df.select_dtypes(include=[np.number]).shape[1]}
• Categorical: {df.select_dtypes(include=['object']).shape[1]}
• Boolean: {df.select_dtypes(include=['bool']).shape[1]}

Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
"""
        ax1.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center', 
                transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.5))
        ax1.set_title("Dataset Information", fontweight='bold')
        ax1.axis('off')
        
        # 2. Missing values heatmap
        ax2 = axes[0, 1]
        missing_data = df.isnull().sum().sort_values(ascending=False)
        if missing_data.sum() > 0:
            missing_data = missing_data[missing_data > 0]
            bars = ax2.bar(range(len(missing_data)), missing_data.values, color=self.colors['warning'])
            ax2.set_title("Missing Values by Column", fontweight='bold')
            ax2.set_xlabel("Columns")
            ax2.set_ylabel("Missing Count")
            ax2.set_xticks(range(len(missing_data)))
            ax2.set_xticklabels(missing_data.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, missing_data.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value}\n({value/len(df)*100:.1f}%)', 
                        ha='center', va='bottom', fontsize=10)
        else:
            ax2.text(0.5, 0.5, "No Missing Values Found!", 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14, fontweight='bold', color=self.colors['success'])
            ax2.set_title("Missing Values Analysis", fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Data types distribution
        ax3 = axes[0, 2]
        dtype_counts = df.dtypes.value_counts()
        wedges, texts, autotexts = ax3.pie(dtype_counts.values, labels=dtype_counts.index, 
                                          autopct='%1.1f%%', startangle=90,
                                          colors=sns.color_palette("husl", len(dtype_counts)))
        ax3.set_title("Data Types Distribution", fontweight='bold')
        
        # 4. Unique values per column
        ax4 = axes[1, 0]
        unique_counts = df.nunique().sort_values(ascending=True)
        bars = ax4.barh(range(len(unique_counts)), unique_counts.values, color=self.colors['primary'])
        ax4.set_title("Unique Values per Column", fontweight='bold')
        ax4.set_xlabel("Number of Unique Values")
        ax4.set_yticks(range(len(unique_counts)))
        ax4.set_yticklabels(unique_counts.index)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, unique_counts.values)):
            ax4.text(bar.get_width() + max(unique_counts) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value}', ha='left', va='center', fontsize=10)
        
        # 5. Correlation matrix preview (numeric columns only)
        ax5 = axes[1, 1]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax5, cbar_kws={'shrink': 0.8})
            ax5.set_title("Correlation Matrix (Numeric Features)", fontweight='bold')
        else:
            ax5.text(0.5, 0.5, "Insufficient numeric columns\nfor correlation analysis", 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title("Correlation Analysis", fontweight='bold')
        
        # 6. Basic statistics summary
        ax6 = axes[1, 2]
        if len(numeric_cols) > 0:
            stats_summary = df[numeric_cols].describe()
            stats_text = f"""
Key Statistics Summary:

Numeric Columns: {len(numeric_cols)}
Mean Values Range: {stats_summary.loc['mean'].min():.2f} - {stats_summary.loc['mean'].max():.2f}
Std Values Range: {stats_summary.loc['std'].min():.2f} - {stats_summary.loc['std'].max():.2f}

Potential Outliers:
• Columns with high std/mean ratio
• Extreme min/max values
• Skewed distributions
"""
            ax6.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.5))
        else:
            ax6.text(0.5, 0.5, "No numeric columns found", ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title("Statistical Summary", fontweight='bold')
        ax6.axis('off')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.output_dir / 'dataset_overview.png', dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def _plot_dataset_overview_interactive(self, df: pd.DataFrame, title: str) -> None:
        """Interactive dataset overview using Plotly."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=["Missing Values", "Data Types", "Unique Values", 
                           "Correlation Heatmap", "Memory Usage", "Dataset Info"],
            specs=[[{"type": "bar"}, {"type": "pie"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "bar"}, {"type": "table"}]]
        )
        
        # Missing values
        missing_data = df.isnull().sum().sort_values(ascending=False)
        if missing_data.sum() > 0:
            missing_data = missing_data[missing_data > 0]
            fig.add_trace(
                go.Bar(x=missing_data.index, y=missing_data.values, name="Missing Values",
                      marker_color=self.colors['warning']),
                row=1, col=1
            )
        
        # Data types pie chart
        dtype_counts = df.dtypes.value_counts()
        fig.add_trace(
            go.Pie(labels=dtype_counts.index, values=dtype_counts.values, name="Data Types"),
            row=1, col=2
        )
        
        # Unique values
        unique_counts = df.nunique().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(x=unique_counts.values, y=unique_counts.index, 
                  orientation='h', name="Unique Values",
                  marker_color=self.colors['primary']),
            row=1, col=3
        )
        
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                          colorscale='RdBu', zmid=0),
                row=2, col=1
            )
        
        fig.update_layout(height=800, title_text=title, showlegend=False)
        fig.show()
    
    def plot_feature_distributions(self, df: pd.DataFrame, 
                                 target_col: Optional[str] = None,
                                 max_features: int = 20) -> None:
        """
        Plot comprehensive feature distributions.
        
        Args:
            df: Input dataframe
            target_col: Target column name for stratified analysis
            max_features: Maximum number of features to plot
        """
        logger.info("Creating feature distribution plots...")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from features if specified
        if target_col:
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            if target_col in categorical_cols:
                categorical_cols.remove(target_col)
        
        # Limit number of features
        numeric_cols = numeric_cols[:max_features]
        categorical_cols = categorical_cols[:max_features//2]
        
        # Plot numeric distributions
        if numeric_cols:
            self._plot_numeric_distributions(df, numeric_cols, target_col)
        
        # Plot categorical distributions
        if categorical_cols:
            self._plot_categorical_distributions(df, categorical_cols, target_col)
    
    def _plot_numeric_distributions(self, df: pd.DataFrame, 
                                  numeric_cols: List[str], 
                                  target_col: Optional[str] = None) -> None:
        """Plot numeric feature distributions."""
        
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle("Numeric Feature Distributions", fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(numeric_cols):
            row, col_idx = divmod(idx, n_cols)
            ax = axes[row, col_idx]
            
            # Plot distribution
            if target_col and target_col in df.columns:
                # Stratified by target
                unique_targets = df[target_col].unique()
                for target_val in unique_targets:
                    subset_data = df[df[target_col] == target_val][col].dropna()
                    ax.hist(subset_data, alpha=0.6, label=f'{target_col}={target_val}',
                           bins=30, density=True)
                ax.legend()
            else:
                # Simple distribution
                ax.hist(df[col].dropna(), bins=30, alpha=0.7, color=self.colors['primary'])
            
            # Add statistics
            mean_val = df[col].mean()
            median_val = df[col].median()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
            
            ax.set_title(f'{col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(numeric_cols), n_rows * n_cols):
            row, col_idx = divmod(idx, n_cols)
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.output_dir / 'numeric_distributions.png', dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def _plot_categorical_distributions(self, df: pd.DataFrame, 
                                      categorical_cols: List[str],
                                      target_col: Optional[str] = None) -> None:
        """Plot categorical feature distributions."""
        
        if not categorical_cols:
            return
        
        n_cols = min(3, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle("Categorical Feature Distributions", fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(categorical_cols):
            row, col_idx = divmod(idx, n_cols)
            ax = axes[row, col_idx]
            
            # Value counts
            value_counts = df[col].value_counts()
            
            if target_col and target_col in df.columns:
                # Stacked bar chart by target
                crosstab = pd.crosstab(df[col], df[target_col])
                crosstab.plot(kind='bar', ax=ax, stacked=True)
                ax.legend(title=target_col)
            else:
                # Simple bar chart
                bars = ax.bar(range(len(value_counts)), value_counts.values, 
                             color=self.colors['primary'])
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                
                # Add value labels
                for bar, value in zip(bars, value_counts.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value}', ha='center', va='bottom')
            
            ax.set_title(f'{col}', fontweight='bold')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(categorical_cols), n_rows * n_cols):
            row, col_idx = divmod(idx, n_cols)
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.output_dir / 'categorical_distributions.png', dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_analysis(self, df: pd.DataFrame, 
                                target_col: Optional[str] = None,
                                method: str = 'pearson') -> None:
        """
        Comprehensive correlation analysis.
        
        Args:
            df: Input dataframe
            target_col: Target column for correlation ranking
            method: Correlation method ('pearson', 'spearman', 'kendall')
        """
        logger.info("Creating correlation analysis plots...")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            logger.warning("Insufficient numeric columns for correlation analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Correlation Analysis ({method.title()})", fontsize=16, fontweight='bold')
        
        # 1. Full correlation heatmap
        ax1 = axes[0, 0]
        corr_matrix = numeric_df.corr(method=method)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax1, cbar_kws={'shrink': 0.8})
        ax1.set_title("Correlation Matrix (Lower Triangle)", fontweight='bold')
        
        # 2. Target correlations (if target specified)
        ax2 = axes[0, 1]
        if target_col and target_col in numeric_df.columns:
            target_corrs = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=True)
            bars = ax2.barh(range(len(target_corrs)), target_corrs.values)
            ax2.set_yticks(range(len(target_corrs)))
            ax2.set_yticklabels(target_corrs.index)
            ax2.set_title(f"Features Correlation with {target_col}", fontweight='bold')
            ax2.set_xlabel(f'|{method.title()} Correlation|')
            
            # Color bars by correlation strength
            for i, (bar, value) in enumerate(zip(bars, target_corrs.values)):
                if value > 0.7:
                    bar.set_color(self.colors['failure'])
                elif value > 0.5:
                    bar.set_color(self.colors['warning'])
                elif value > 0.3:
                    bar.set_color(self.colors['success'])
                else:
                    bar.set_color(self.colors['neutral'])
        else:
            ax2.text(0.5, 0.5, "No target column specified", ha='center', va='center', 
                    transform=ax2.transAxes)
            ax2.set_title("Target Correlations", fontweight='bold')
        
        # 3. High correlation pairs
        ax3 = axes[1, 0]
        # Find high correlation pairs (excluding self-correlations)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            pair_labels = [f"{pair[0]} vs {pair[1]}" for pair in high_corr_pairs[:10]]
            pair_values = [abs(pair[2]) for pair in high_corr_pairs[:10]]
            
            bars = ax3.barh(range(len(pair_labels)), pair_values, color=self.colors['warning'])
            ax3.set_yticks(range(len(pair_labels)))
            ax3.set_yticklabels(pair_labels)
            ax3.set_title("High Correlation Pairs (|r| > 0.7)", fontweight='bold')
            ax3.set_xlabel(f'|{method.title()} Correlation|')
            
            # Add value labels
            for bar, value in zip(bars, pair_values):
                ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', ha='left', va='center')
        else:
            ax3.text(0.5, 0.5, "No high correlation pairs found\n(threshold: |r| > 0.7)", 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("High Correlation Pairs", fontweight='bold')
        
        # 4. Correlation distribution
        ax4 = axes[1, 1]
        # Get all correlation values (upper triangle, excluding diagonal)
        corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        ax4.hist(corr_values, bins=30, alpha=0.7, color=self.colors['primary'], edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax4.set_title("Distribution of Correlation Values", fontweight='bold')
        ax4.set_xlabel(f'{method.title()} Correlation')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics
        mean_corr = np.mean(np.abs(corr_values))
        ax4.text(0.05, 0.95, f'Mean |correlation|: {mean_corr:.3f}', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.output_dir / f'correlation_analysis_{method}.png', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_target_analysis(self, df: pd.DataFrame, target_col: str) -> None:
        """
        Comprehensive target variable analysis.
        
        Args:
            df: Input dataframe
            target_col: Target column name
        """
        logger.info(f"Creating target analysis plots for {target_col}...")
        
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in dataframe")
            return
        
        # Determine if target is numeric or categorical
        is_numeric_target = pd.api.types.is_numeric_dtype(df[target_col])
        
        if is_numeric_target:
            self._plot_numeric_target_analysis(df, target_col)
        else:
            self._plot_categorical_target_analysis(df, target_col)
    
    def _plot_categorical_target_analysis(self, df: pd.DataFrame, target_col: str) -> None:
        """Target analysis for categorical variables (classification)."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Target Analysis: {target_col} (Classification)", fontsize=16, fontweight='bold')
        
        # 1. Target distribution
        ax1 = axes[0, 0]
        target_counts = df[target_col].value_counts()
        bars = ax1.bar(target_counts.index, target_counts.values, 
                      color=[self.colors['normal'] if val == 0 else self.colors['failure'] 
                            for val in target_counts.index])
        ax1.set_title("Target Distribution", fontweight='bold')
        ax1.set_xlabel(target_col)
        ax1.set_ylabel('Count')
        
        # Add percentage labels
        total = len(df)
        for bar, value in zip(bars, target_counts.values):
            percentage = value / total * 100
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                    f'{value}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        # 2. Target balance pie chart
        ax2 = axes[0, 1]
        colors = [self.colors['normal'], self.colors['failure']] if len(target_counts) == 2 else None
        wedges, texts, autotexts = ax2.pie(target_counts.values, labels=target_counts.index,
                                          autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title("Target Balance", fontweight='bold')
        
        # 3. Class imbalance metrics
        ax3 = axes[0, 2]
        imbalance_ratio = target_counts.max() / target_counts.min() if len(target_counts) > 1 else 1
        minority_class_pct = (target_counts.min() / total) * 100 if len(target_counts) > 1 else 50
        
        metrics_text = f"""
Class Imbalance Metrics:

Total Samples: {total:,}
Classes: {len(target_counts)}

Imbalance Ratio: {imbalance_ratio:.2f}:1
Minority Class: {minority_class_pct:.1f}%

Recommendation:
{'⚠️  Consider resampling' if imbalance_ratio > 3 else '✅  Balanced dataset'}
{'⚠️  Very imbalanced!' if minority_class_pct < 10 else ''}
"""
        ax3.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                transform=ax3.transAxes, 
                bbox=dict(boxstyle="round", 
                         facecolor='lightcoral' if imbalance_ratio > 3 else 'lightgreen', 
                         alpha=0.5))
        ax3.set_title("Imbalance Assessment", fontweight='bold')
        ax3.axis('off')
        
        # 4. Feature distributions by target (top numeric features)
        ax4 = axes[1, 0]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if numeric_cols:
            # Select top correlated feature with target
            correlations = []
            for col in numeric_cols[:5]:  # Check top 5 numeric columns
                if df[col].dtype in ['int64', 'float64']:
                    # Create temporary numeric encoding for correlation
                    target_encoded = pd.factorize(df[target_col])[0]
                    corr = np.corrcoef(df[col].fillna(df[col].mean()), target_encoded)[0, 1]
                    correlations.append((col, abs(corr)))
            
            if correlations:
                best_feature = max(correlations, key=lambda x: x[1])[0]
                
                # Box plot of best feature by target
                df.boxplot(column=best_feature, by=target_col, ax=ax4)
                ax4.set_title(f"{best_feature} by {target_col}", fontweight='bold')
                ax4.set_xlabel(target_col)
                ax4.set_ylabel(best_feature)
        else:
            ax4.text(0.5, 0.5, "No numeric features available", ha='center', va='center', 
                    transform=ax4.transAxes)
            ax4.set_title("Feature Analysis", fontweight='bold')
        
        # 5. Sample distribution over time (if UDI column exists)
        ax5 = axes[1, 1]
        if 'UDI' in df.columns:
            # Group by UDI ranges and show target distribution
            df['UDI_bin'] = pd.cut(df['UDI'], bins=10)
            udi_target_dist = df.groupby('UDI_bin')[target_col].value_counts().unstack(fill_value=0)
            udi_target_dist.plot(kind='bar', stacked=True, ax=ax5, 
                               color=[self.colors['normal'], self.colors['failure']])
            ax5.set_title("Target Distribution Over Time (UDI)", fontweight='bold')
            ax5.set_xlabel("UDI Bins")
            ax5.set_ylabel("Count")
            ax5.legend(title=target_col)
            plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax5.text(0.5, 0.5, "No UDI column found", ha='center', va='center', 
                    transform=ax5.transAxes)
            ax5.set_title("Temporal Analysis", fontweight='bold')
        
        # 6. Statistical summary by target
        ax6 = axes[1, 2]
        if numeric_cols:
            summary_stats = df.groupby(target_col)[numeric_cols[:3]].agg(['mean', 'std']).round(3)
            summary_text = f"Statistical Summary by {target_col}:\n\n"
            
            for target_val in df[target_col].unique():
                summary_text += f"{target_col} = {target_val}:\n"
                for col in numeric_cols[:3]:
                    if col in summary_stats.columns.levels[0]:
                        mean_val = summary_stats.loc[target_val, (col, 'mean')]
                        std_val = summary_stats.loc[target_val, (col, 'std')]
                        summary_text += f"  {col}: {mean_val:.2f} ± {std_val:.2f}\n"
                summary_text += "\n"
            
            ax6.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
                    transform=ax6.transAxes, 
                    bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.5))
        else:
            ax6.text(0.5, 0.5, "No numeric features for summary", ha='center', va='center', 
                    transform=ax6.transAxes)
        ax6.set_title("Statistical Summary", fontweight='bold')
        ax6.axis('off')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.output_dir / f'target_analysis_{target_col}.png', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def _plot_numeric_target_analysis(self, df: pd.DataFrame, target_col: str) -> None:
        """Target analysis for numeric variables (regression)."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Target Analysis: {target_col} (Regression)", fontsize=16, fontweight='bold')
        
        target_data = df[target_col].dropna()
        
        # 1. Target distribution
        ax1 = axes[0, 0]
        ax1.hist(target_data, bins=50, alpha=0.7, color=self.colors['primary'], edgecolor='black')
        ax1.set_title("Target Distribution", fontweight='bold')
        ax1.set_xlabel(target_col)
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = target_data.mean()
        median_val = target_data.median()
        ax1.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
        ax1.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
        ax1.legend()
        
        # 2. Q-Q plot for normality assessment
        ax2 = axes[0, 1]
        stats.probplot(target_data, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normality Check)", fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Box plot and outlier analysis
        ax3 = axes[0, 2]
        box_plot = ax3.boxplot(target_data, patch_artist=True)
        box_plot['boxes'][0].set_facecolor(self.colors['primary'])
        ax3.set_title("Box Plot & Outliers", fontweight='bold')
        ax3.set_ylabel(target_col)
        ax3.grid(True, alpha=0.3)
        
        # Calculate outlier statistics
        Q1 = target_data.quantile(0.25)
        Q3 = target_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = target_data[(target_data < Q1 - 1.5*IQR) | (target_data > Q3 + 1.5*IQR)]
        
        ax3.text(0.02, 0.98, f'Outliers: {len(outliers)} ({len(outliers)/len(target_data)*100:.1f}%)', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.5))
        
        # 4. Statistical tests
        ax4 = axes[1, 0]
        # Normality tests
        shapiro_stat, shapiro_p = shapiro(target_data.sample(min(5000, len(target_data))))
        jb_stat, jb_p = jarque_bera(target_data)
        
        test_results = f"""
Statistical Tests:

Descriptive Statistics:
• Mean: {target_data.mean():.3f}
• Std: {target_data.std():.3f}
• Skewness: {target_data.skew():.3f}
• Kurtosis: {target_data.kurtosis():.3f}

Normality Tests:
• Shapiro-Wilk: p={shapiro_p:.4f}
• Jarque-Bera: p={jb_p:.4f}

{'✅ Approximately normal' if min(shapiro_p, jb_p) > 0.05 else '⚠️  Not normal distribution'}
"""
        ax4.text(0.1, 0.9, test_results, fontsize=11, verticalalignment='top',
                transform=ax4.transAxes, 
                bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.5))
        ax4.set_title("Statistical Tests", fontweight='bold')
        ax4.axis('off')
        
        # 5. Correlation with numeric features
        ax5 = axes[1, 1]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if len(numeric_cols) > 0:
            correlations = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
            correlations = correlations.abs().sort_values(ascending=True)
            
            bars = ax5.barh(range(len(correlations)), correlations.values)
            ax5.set_yticks(range(len(correlations)))
            ax5.set_yticklabels(correlations.index)
            ax5.set_title(f"Feature Correlations with {target_col}", fontweight='bold')
            ax5.set_xlabel('|Pearson Correlation|')
            ax5.grid(True, alpha=0.3)
            
            # Color bars by correlation strength
            for bar, value in zip(bars, correlations.values):
                if value > 0.7:
                    bar.set_color(self.colors['failure'])
                elif value > 0.5:
                    bar.set_color(self.colors['warning'])
                elif value > 0.3:
                    bar.set_color(self.colors['success'])
                else:
                    bar.set_color(self.colors['neutral'])
        else:
            ax5.text(0.5, 0.5, "No other numeric features", ha='center', va='center', 
                    transform=ax5.transAxes)
            ax5.set_title("Feature Correlations", fontweight='bold')
        
        # 6. Target vs. best correlated feature
        ax6 = axes[1, 2]
        if len(numeric_cols) > 0:
            # Find best correlated feature
            correlations = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
            best_feature = correlations.abs().idxmax()
            best_corr = correlations[best_feature]
            
            ax6.scatter(df[best_feature], df[target_col], alpha=0.6, color=self.colors['primary'])
            ax6.set_xlabel(best_feature)
            ax6.set_ylabel(target_col)
            ax6.set_title(f"{target_col} vs {best_feature}\n(r={best_corr:.3f})", fontweight='bold')
            ax6.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(df[best_feature].dropna(), df[target_col].dropna(), 1)
            p = np.poly1d(z)
            ax6.plot(df[best_feature], p(df[best_feature]), "r--", alpha=0.8)
        else:
            ax6.text(0.5, 0.5, "No features for scatter plot", ha='center', va='center', 
                    transform=ax6.transAxes)
            ax6.set_title("Feature Relationships", fontweight='bold')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.output_dir / f'target_analysis_{target_col}.png', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_missing_data_analysis(self, df: pd.DataFrame) -> None:
        """
        Comprehensive missing data analysis.
        
        Args:
            df: Input dataframe
        """
        logger.info("Creating missing data analysis plots...")
        
        # Calculate missing data
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_percent = (missing_data / len(df) * 100).round(2)
        
        if missing_data.sum() == 0:
            print("🎉 No missing data found in the dataset!")
            return
        
        # Filter columns with missing data
        missing_cols = missing_data[missing_data > 0]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Missing Data Analysis", fontsize=16, fontweight='bold')
        
        # 1. Missing data counts
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(missing_cols)), missing_cols.values, color=self.colors['warning'])
        ax1.set_title("Missing Data Count by Column", fontweight='bold')
        ax1.set_xlabel("Columns")
        ax1.set_ylabel("Missing Count")
        ax1.set_xticks(range(len(missing_cols)))
        ax1.set_xticklabels(missing_cols.index, rotation=45, ha='right')
        
        # Add percentage labels
        for bar, count, pct in zip(bars, missing_cols.values, missing_percent[missing_cols.index]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(missing_cols)*0.01,
                    f'{count}\n({pct}%)', ha='center', va='bottom', fontsize=9)
        
        # 2. Missing data heatmap
        ax2 = axes[0, 1]
        if len(missing_cols) > 1:
            # Create missing data matrix
            missing_matrix = df[missing_cols.index].isnull()
            sns.heatmap(missing_matrix, cbar=True, ax=ax2, cmap='viridis',
                       yticklabels=False, xticklabels=True)
            ax2.set_title("Missing Data Pattern", fontweight='bold')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax2.text(0.5, 0.5, "Only one column with missing data", ha='center', va='center', 
                    transform=ax2.transAxes)
            ax2.set_title("Missing Data Pattern", fontweight='bold')
        
        # 3. Missing data statistics
        ax3 = axes[1, 0]
        total_cells = df.shape[0] * df.shape[1]
        total_missing = missing_data.sum()
        
        stats_text = f"""
Missing Data Summary:

Total Cells: {total_cells:,}
Missing Cells: {total_missing:,}
Overall Missing Rate: {(total_missing/total_cells)*100:.2f}%

Columns Affected: {len(missing_cols)}/{len(df.columns)}
Rows with Any Missing: {df.isnull().any(axis=1).sum():,}
Complete Rows: {(~df.isnull().any(axis=1)).sum():,}

Most Missing Column: {missing_cols.index[0]} ({missing_percent[missing_cols.index[0]]:.1f}%)
"""
        ax3.text(0.1, 0.9, stats_text, fontsize=11, verticalalignment='top',
                transform=ax3.transAxes, 
                bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.7))
        ax3.set_title("Missing Data Statistics", fontweight='bold')
        ax3.axis('off')
        
        # 4. Missing data co-occurrence
        ax4 = axes[1, 1]
        if len(missing_cols) > 1:
            # Calculate correlation between missing patterns
            missing_corr = df[missing_cols.index].isnull().corr()
            sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0, ax=ax4,
                       square=True, cbar_kws={'shrink': 0.8})
            ax4.set_title("Missing Data Co-occurrence", fontweight='bold')
        else:
            ax4.text(0.5, 0.5, "Single column - no co-occurrence analysis", 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Missing Data Co-occurrence", fontweight='bold')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.output_dir / 'missing_data_analysis.png', dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_outlier_analysis(self, df: pd.DataFrame, method: str = 'iqr') -> None:
        """
        Comprehensive outlier analysis.
        
        Args:
            df: Input dataframe
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
        """
        logger.info(f"Creating outlier analysis plots using {method} method...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for outlier analysis")
            return
        
        # Limit to first 12 columns for visualization
        numeric_cols = numeric_cols[:12]
        
        # Calculate outliers
        outlier_results = {}
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = df[col][z_scores > 3]
            
            outlier_results[col] = len(outliers)
        
        # Create plots
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows + 1, n_cols, figsize=(n_cols * 4, (n_rows + 1) * 3))
        fig.suptitle(f"Outlier Analysis ({method.upper()} Method)", fontsize=16, fontweight='bold')
        
        # Summary plot (first row)
        ax_summary = plt.subplot2grid((n_rows + 1, n_cols), (0, 0), colspan=n_cols)
        outlier_counts = list(outlier_results.values())
        outlier_percentages = [(count/len(df))*100 for count in outlier_counts]
        
        bars = ax_summary.bar(range(len(numeric_cols)), outlier_counts, color=self.colors['warning'])
        ax_summary.set_title(f"Outlier Count by Column ({method.upper()})", fontweight='bold')
        ax_summary.set_xlabel("Columns")
        ax_summary.set_ylabel("Outlier Count")
        ax_summary.set_xticks(range(len(numeric_cols)))
        ax_summary.set_xticklabels(numeric_cols, rotation=45, ha='right')
        
        # Add percentage labels
        for bar, count, pct in zip(bars, outlier_counts, outlier_percentages):
            ax_summary.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(outlier_counts)*0.01,
                           f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
        
        # Individual box plots (remaining rows)
        for idx, col in enumerate(numeric_cols):
            row = (idx // n_cols) + 1
            col_idx = idx % n_cols
            ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx + n_cols]
            
            # Box plot
            box_plot = ax.boxplot(df[col].dropna(), patch_artist=True)
            box_plot['boxes'][0].set_facecolor(self.colors['primary'])
            
            # Highlight outliers
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = df[col][z_scores > 3]
            
            if len(outliers) > 0:
                ax.scatter([1] * len(outliers), outliers, color=self.colors['failure'], 
                          alpha=0.6, s=30, label=f'Outliers ({len(outliers)})')
                ax.legend()
            
            ax.set_title(f'{col}', fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        total_plots = (n_rows + 1) * n_cols
        for idx in range(len(numeric_cols) + n_cols, total_plots):
            row = idx // n_cols
            col_idx = idx % n_cols
            if row < len(axes) and col_idx < len(axes[0]):
                axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.output_dir / f'outlier_analysis_{method}.png', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def generate_eda_report(self, df: pd.DataFrame, 
                          target_col: Optional[str] = None,
                          title: str = "AI4I Predictive Maintenance EDA Report") -> None:
        """
        Generate a comprehensive EDA report with all visualizations.
        
        Args:
            df: Input dataframe
            target_col: Target column name
            title: Report title
        """
        logger.info("Generating comprehensive EDA report...")
        
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}\n")
        
        # 1. Dataset Overview
        print("📊 DATASET OVERVIEW")
        print("-" * 40)
        self.plot_dataset_overview(df, title)
        
        # 2. Feature Distributions
        print("\n📈 FEATURE DISTRIBUTIONS")
        print("-" * 40)
        self.plot_feature_distributions(df, target_col)
        
        # 3. Correlation Analysis
        print("\n🔗 CORRELATION ANALYSIS")
        print("-" * 40)
        self.plot_correlation_analysis(df, target_col)
        
        # 4. Target Analysis (if specified)
        if target_col:
            print(f"\n🎯 TARGET ANALYSIS ({target_col})")
            print("-" * 40)
            self.plot_target_analysis(df, target_col)
        
        # 5. Missing Data Analysis
        print("\n❓ MISSING DATA ANALYSIS")
        print("-" * 40)
        self.plot_missing_data_analysis(df)
        
        # 6. Outlier Analysis
        print("\n🚨 OUTLIER ANALYSIS")
        print("-" * 40)
        self.plot_outlier_analysis(df, method='iqr')
        
        print(f"\n{'='*60}")
        print("EDA REPORT COMPLETED")
        print(f"Plots saved to: {self.output_dir}")
        print(f"{'='*60}\n")


# Convenience functions for quick EDA
def quick_eda(df: pd.DataFrame, 
              target_col: Optional[str] = None,
              save_plots: bool = True,
              interactive: bool = False) -> None:
    """
    Quick EDA analysis for AI4I dataset.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        save_plots: Whether to save plots
        interactive: Use interactive plots
    """
    plotter = EDAPlotter(save_plots=save_plots, interactive=interactive)
    plotter.generate_eda_report(df, target_col, "AI4I Quick EDA Analysis")


def ai4i_specific_eda(data_path: str, 
                     target_col: str = 'Machine failure',
                     save_plots: bool = True) -> None:
    """
    AI4I dataset-specific EDA analysis.
    
    Args:
        data_path: Path to AI4I dataset
        target_col: Target column name
        save_plots: Whether to save plots
    """
    logger.info(f"Loading AI4I dataset from {data_path}...")
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Dataset loaded successfully: {df.shape}")
        
        plotter = EDAPlotter(save_plots=save_plots)
        plotter.generate_eda_report(df, target_col, "AI4I 2020 Predictive Maintenance EDA")
        
        # Additional AI4I-specific analysis
        logger.info("Performing AI4I-specific analysis...")
        
        # Process failure types if available
        failure_cols = [col for col in df.columns if 'failure' in col.lower() and col != target_col]
        if failure_cols:
            fig, ax = plt.subplots(figsize=(12, 6))
            failure_counts = df[failure_cols].sum().sort_values(ascending=True)
            bars = ax.barh(range(len(failure_counts)), failure_counts.values)
            ax.set_yticks(range(len(failure_counts)))
            ax.set_yticklabels(failure_counts.index)
            ax.set_title("Failure Types Distribution", fontweight='bold', fontsize=14)
            ax.set_xlabel("Count")
            
            # Add value labels
            for bar, value in zip(bars, failure_counts.values):
                ax.text(bar.get_width() + max(failure_counts)*0.01, 
                       bar.get_y() + bar.get_height()/2,
                       f'{value}', ha='left', va='center')
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('reports/figures/ai4i_failure_types.png', dpi=300, bbox_inches='tight')
            plt.show()
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")


# Testing function
def test_eda_plots():
    """Test EDA plotting functionality."""
    
    # Create sample data similar to AI4I structure
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'UDI': range(1, n_samples + 1),
        'Product ID': [f'L{i}' if i % 3 == 0 else f'M{i}' if i % 3 == 1 else f'H{i}' 
                      for i in range(n_samples)],
        'Type': np.random.choice(['L', 'M', 'H'], n_samples),
        'Air temperature [K]': np.random.normal(300, 2, n_samples),
        'Process temperature [K]': np.random.normal(310, 1.5, n_samples),
        'Rotational speed [rpm]': np.random.normal(1500, 100, n_samples),
        'Torque [Nm]': np.random.normal(40, 10, n_samples),
        'Tool wear [min]': np.random.exponential(100, n_samples),
        'Machine failure': np.random.choice([0, 1], n_samples, p=[0.97, 0.03])
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Testing EDA plots with sample data...")
    plotter = EDAPlotter(save_plots=False)
    
    # Test individual plot functions
    plotter.plot_dataset_overview(df)
    plotter.plot_feature_distributions(df, 'Machine failure')
    plotter.plot_correlation_analysis(df, 'Machine failure')
    plotter.plot_target_analysis(df, 'Machine failure')
    
    print("EDA plots testing completed successfully!")


if __name__ == "__main__":
    test_eda_plots()
