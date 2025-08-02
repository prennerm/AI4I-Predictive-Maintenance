"""
Feature Selection Module for AI4I Predictive Maintenance Project

This module provides comprehensive feature selection functionality including:
- Filter methods (correlation, variance, univariate tests)
- Wrapper methods (RFE, sequential selection)
- Embedded methods (model-based importance, regularization)
- Hybrid methods (multi-criteria selection)
- Pipeline integration for automated feature selection

Author: AI4I Team
Date: August 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, VarianceThreshold, 
    RFE, RFECV, SelectFromModel, 
    f_classif, chi2, mutual_info_classif,
    SequentialFeatureSelector
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Statistical imports
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)

class FeatureSelector:
    """
    Comprehensive feature selection class for the AI4I dataset.
    
    This class implements various feature selection methods:
    - Filter methods: Statistical tests, correlation, variance
    - Wrapper methods: RFE, sequential selection
    - Embedded methods: Model-based importance, regularization
    - Hybrid methods: Multi-criteria combinations
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the FeatureSelector.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.selection_results = {}
        self.scaler = StandardScaler()
        
    def select_by_correlation(self, X: pd.DataFrame, threshold: float = 0.9,
                            method: str = 'pearson') -> List[str]:
        """
        Remove highly correlated features using correlation analysis.
        
        Args:
            X (pd.DataFrame): Feature matrix
            threshold (float): Correlation threshold (0-1)
            method (str): Correlation method ('pearson', 'spearman')
            
        Returns:
            List[str]: Selected feature names
        """
        logger.info(f"Selecting features by correlation (threshold={threshold}, method={method})")
        
        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = X.corr().abs()
        elif method == 'spearman':
            corr_matrix = X.corr(method='spearman').abs()
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        
        # Find pairs of highly correlated features
        upper_tri = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        )
        
        # Identify features to drop
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > threshold)]
        
        selected_features = [col for col in X.columns if col not in to_drop]
        
        self.selection_results['correlation'] = {
            'method': method,
            'threshold': threshold,
            'original_features': len(X.columns),
            'selected_features': len(selected_features),
            'dropped_features': to_drop,
            'selected_names': selected_features
        }
        
        logger.info(f"Correlation selection: {len(X.columns)} → {len(selected_features)} features")
        return selected_features
    
    def select_by_variance(self, X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """
        Remove low-variance features.
        
        Args:
            X (pd.DataFrame): Feature matrix
            threshold (float): Variance threshold
            
        Returns:
            List[str]: Selected feature names
        """
        logger.info(f"Selecting features by variance (threshold={threshold})")
        
        # Calculate variances
        variances = X.var()
        
        # Select features above threshold
        selected_features = variances[variances > threshold].index.tolist()
        
        self.selection_results['variance'] = {
            'threshold': threshold,
            'original_features': len(X.columns),
            'selected_features': len(selected_features),
            'feature_variances': variances.to_dict(),
            'selected_names': selected_features
        }
        
        logger.info(f"Variance selection: {len(X.columns)} → {len(selected_features)} features")
        return selected_features
    
    def select_by_univariate(self, X: pd.DataFrame, y: pd.Series, 
                           score_func: str = 'f_classif', k: int = 20) -> List[str]:
        """
        Select features using univariate statistical tests.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            score_func (str): Scoring function ('f_classif', 'chi2', 'mutual_info')
            k (int): Number of features to select
            
        Returns:
            List[str]: Selected feature names
        """
        logger.info(f"Selecting features by univariate test ({score_func}, k={k})")
        
        # Choose scoring function
        if score_func == 'f_classif':
            scoring_function = f_classif
        elif score_func == 'chi2':
            scoring_function = chi2
            # Ensure non-negative values for chi2
            X = X - X.min() + 1e-6
        elif score_func == 'mutual_info':
            scoring_function = mutual_info_classif
        else:
            raise ValueError(f"Unsupported scoring function: {score_func}")
        
        # Perform selection
        selector = SelectKBest(score_func=scoring_function, k=min(k, X.shape[1]))
        selector.fit(X, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Get scores
        scores = selector.scores_ if hasattr(selector, 'scores_') else np.zeros(len(X.columns))
        feature_scores = dict(zip(X.columns, scores))
        
        self.selection_results['univariate'] = {
            'score_func': score_func,
            'k': k,
            'original_features': len(X.columns),
            'selected_features': len(selected_features),
            'feature_scores': feature_scores,
            'selected_names': selected_features
        }
        
        logger.info(f"Univariate selection: {len(X.columns)} → {len(selected_features)} features")
        return selected_features
    
    def select_by_rfe(self, X: pd.DataFrame, y: pd.Series, 
                     estimator: Any = None, n_features: int = 15,
                     cv: int = 5) -> List[str]:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            estimator: Estimator to use for RFE
            n_features (int): Number of features to select
            cv (int): Cross-validation folds (if 0, use RFE instead of RFECV)
            
        Returns:
            List[str]: Selected feature names
        """
        logger.info(f"Selecting features by RFE (n_features={n_features}, cv={cv})")
        
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        # Scale features for linear models
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Use RFECV if cv > 0, otherwise RFE
        if cv > 0:
            selector = RFECV(
                estimator=estimator, 
                step=1, 
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
                scoring='f1',
                n_jobs=-1
            )
        else:
            selector = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
        
        selector.fit(X_scaled, y)
        
        # Get selected features
        selected_mask = selector.support_
        selected_features = X.columns[selected_mask].tolist()
        
        # Get rankings
        rankings = selector.ranking_ if hasattr(selector, 'ranking_') else np.ones(len(X.columns))
        feature_rankings = dict(zip(X.columns, rankings))
        
        self.selection_results['rfe'] = {
            'estimator': type(estimator).__name__,
            'n_features_requested': n_features,
            'n_features_selected': len(selected_features),
            'cv': cv,
            'feature_rankings': feature_rankings,
            'selected_names': selected_features
        }
        
        logger.info(f"RFE selection: {len(X.columns)} → {len(selected_features)} features")
        return selected_features
    
    def select_by_importance(self, X: pd.DataFrame, y: pd.Series,
                           model: str = 'random_forest', top_k: int = 20) -> List[str]:
        """
        Select features based on model feature importance.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            model (str): Model type ('random_forest', 'extra_trees', 'decision_tree')
            top_k (int): Number of top features to select
            
        Returns:
            List[str]: Selected feature names
        """
        logger.info(f"Selecting features by importance ({model}, top_k={top_k})")
        
        # Choose model
        if model == 'random_forest':
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif model == 'extra_trees':
            estimator = ExtraTreesClassifier(n_estimators=100, random_state=self.random_state)
        elif model == 'decision_tree':
            estimator = DecisionTreeClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        # Fit model and get importances
        estimator.fit(X, y)
        importances = estimator.feature_importances_
        
        # Create importance DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top k features
        selected_features = importance_df.head(top_k)['feature'].tolist()
        
        self.selection_results['importance'] = {
            'model': model,
            'top_k': top_k,
            'original_features': len(X.columns),
            'selected_features': len(selected_features),
            'feature_importances': dict(zip(X.columns, importances)),
            'selected_names': selected_features
        }
        
        logger.info(f"Importance selection: {len(X.columns)} → {len(selected_features)} features")
        return selected_features
    
    def select_by_lasso(self, X: pd.DataFrame, y: pd.Series, 
                       alpha: Optional[float] = None) -> List[str]:
        """
        Select features using LASSO regularization.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            alpha (float): Regularization strength (if None, use CV)
            
        Returns:
            List[str]: Selected feature names
        """
        logger.info(f"Selecting features by LASSO (alpha={alpha})")
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Use LASSO with cross-validation to find optimal alpha
        if alpha is None:
            lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=1000)
        else:
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=alpha, random_state=self.random_state, max_iter=1000)
        
        lasso.fit(X_scaled, y)
        
        # Get selected features (non-zero coefficients)
        selected_mask = lasso.coef_ != 0
        selected_features = X.columns[selected_mask].tolist()
        
        # Get coefficients
        feature_coefficients = dict(zip(X.columns, lasso.coef_))
        
        self.selection_results['lasso'] = {
            'alpha_used': lasso.alpha_ if hasattr(lasso, 'alpha_') else alpha,
            'original_features': len(X.columns),
            'selected_features': len(selected_features),
            'feature_coefficients': feature_coefficients,
            'selected_names': selected_features
        }
        
        logger.info(f"LASSO selection: {len(X.columns)} → {len(selected_features)} features")
        return selected_features
    
    def select_by_sequential(self, X: pd.DataFrame, y: pd.Series,
                           direction: str = 'forward', n_features: int = 15,
                           estimator: Any = None) -> List[str]:
        """
        Select features using Sequential Feature Selection.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            direction (str): Direction ('forward', 'backward')
            n_features (int): Number of features to select
            estimator: Estimator to use
            
        Returns:
            List[str]: Selected feature names
        """
        logger.info(f"Selecting features by sequential selection ({direction}, n_features={n_features})")
        
        if estimator is None:
            estimator = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Perform sequential selection
        selector = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=min(n_features, X.shape[1]),
            direction=direction,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        
        selector.fit(X_scaled, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        self.selection_results['sequential'] = {
            'direction': direction,
            'estimator': type(estimator).__name__,
            'n_features_requested': n_features,
            'n_features_selected': len(selected_features),
            'selected_names': selected_features
        }
        
        logger.info(f"Sequential selection: {len(X.columns)} → {len(selected_features)} features")
        return selected_features
    
    def select_best_features(self, X: pd.DataFrame, y: pd.Series,
                           method: str = 'hybrid', target_size: Optional[int] = None) -> List[str]:
        """
        Select the best features using a hybrid approach.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): Selection method ('hybrid', 'aggressive', 'conservative')
            target_size (int): Target number of features
            
        Returns:
            List[str]: Selected feature names
        """
        logger.info(f"Selecting best features using {method} method")
        
        if target_size is None:
            # Heuristic: √n_samples or n_samples/10 (whichever is smaller)
            target_size = min(int(np.sqrt(len(X))), len(X)//10, X.shape[1]//2)
            target_size = max(target_size, 5)  # Minimum 5 features
        
        if method == 'hybrid':
            # Multi-stage selection
            logger.info("Stage 1: Correlation filtering")
            stage1_features = self.select_by_correlation(X, threshold=0.9)
            X_stage1 = X[stage1_features]
            
            logger.info("Stage 2: Variance filtering")
            stage2_features = self.select_by_variance(X_stage1, threshold=0.01)
            X_stage2 = X_stage1[stage2_features]
            
            logger.info("Stage 3: Importance-based selection")
            stage3_features = self.select_by_importance(X_stage2, y, top_k=min(target_size*2, len(stage2_features)))
            X_stage3 = X_stage2[stage3_features]
            
            logger.info("Stage 4: Final RFE selection")
            final_features = self.select_by_rfe(X_stage3, y, n_features=min(target_size, len(stage3_features)))
            
        elif method == 'aggressive':
            # Aggressive reduction
            corr_features = self.select_by_correlation(X, threshold=0.7)
            X_corr = X[corr_features]
            final_features = self.select_by_importance(X_corr, y, top_k=target_size)
            
        elif method == 'conservative':
            # Conservative selection
            var_features = self.select_by_variance(X, threshold=0.001)
            X_var = X[var_features]
            final_features = self.select_by_univariate(X_var, y, k=target_size)
            
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        self.selection_results['best_features'] = {
            'method': method,
            'target_size': target_size,
            'final_size': len(final_features),
            'selected_names': final_features
        }
        
        logger.info(f"Best features selection: {len(X.columns)} → {len(final_features)} features")
        return final_features
    
    def evaluate_feature_sets(self, X: pd.DataFrame, y: pd.Series,
                            feature_sets: Dict[str, List[str]], 
                            estimator: Any = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate different feature sets using cross-validation.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            feature_sets (Dict): Dictionary of feature sets to evaluate
            estimator: Estimator to use for evaluation
            
        Returns:
            Dict: Evaluation results for each feature set
        """
        logger.info(f"Evaluating {len(feature_sets)} feature sets")
        
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        results = {}
        
        for set_name, features in feature_sets.items():
            if len(features) == 0:
                logger.warning(f"Skipping empty feature set: {set_name}")
                continue
                
            X_subset = X[features]
            
            # Cross-validation
            cv_scores = cross_val_score(
                estimator, X_subset, y, 
                cv=5, scoring='f1', n_jobs=-1
            )
            
            results[set_name] = {
                'n_features': len(features),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'features': features
            }
            
            logger.info(f"{set_name}: {len(features)} features, F1={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        return results
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all selection results.
        
        Returns:
            Dict: Summary of selection results
        """
        summary = {
            'methods_used': list(self.selection_results.keys()),
            'results': self.selection_results
        }
        
        return summary
    
    def save_selection_results(self, output_dir: str, filename: str = "feature_selection_results.json") -> None:
        """
        Save feature selection results to disk.
        
        Args:
            output_dir (str): Output directory
            filename (str): Output filename
        """
        import json
        
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        results_copy = {}
        for method, results in self.selection_results.items():
            results_copy[method] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    results_copy[method][key] = value.tolist()
                elif isinstance(value, np.integer):
                    results_copy[method][key] = int(value)
                elif isinstance(value, np.floating):
                    results_copy[method][key] = float(value)
                else:
                    results_copy[method][key] = value
        
        with open(output_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        logger.info(f"Selection results saved to: {output_path}")


def create_comparison_feature_sets(X: pd.DataFrame, y: pd.Series, 
                                 output_dir: str) -> Dict[str, List[str]]:
    """
    Create multiple feature sets for comparison experiments.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        output_dir (str): Directory to save results
        
    Returns:
        Dict[str, List[str]]: Dictionary of feature sets
    """
    selector = FeatureSelector()
    
    feature_sets = {}
    
    # Create different selection strategies
    feature_sets['all_features'] = X.columns.tolist()
    feature_sets['correlation_filtered'] = selector.select_by_correlation(X, threshold=0.9)
    feature_sets['variance_filtered'] = selector.select_by_variance(X, threshold=0.01)
    feature_sets['univariate_top20'] = selector.select_by_univariate(X, y, k=20)
    feature_sets['importance_top15'] = selector.select_by_importance(X, y, top_k=15)
    feature_sets['rfe_top15'] = selector.select_by_rfe(X, y, n_features=15)
    feature_sets['lasso_selected'] = selector.select_by_lasso(X, y)
    feature_sets['hybrid_best'] = selector.select_best_features(X, y, method='hybrid')
    feature_sets['aggressive_best'] = selector.select_best_features(X, y, method='aggressive')
    
    # Evaluate all feature sets
    evaluation_results = selector.evaluate_feature_sets(X, y, feature_sets)
    
    # Save results
    selector.save_selection_results(output_dir)
    
    # Save evaluation results
    import json
    eval_path = Path(output_dir) / "feature_set_evaluations.json"
    with open(eval_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    logger.info(f"Created {len(feature_sets)} feature sets for comparison")
    
    return feature_sets


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from features.feature_engineering import FeatureEngineer
    from utils.data_preprocessing import preprocess_ai4i_data
    
    # Load and preprocess data
    summary = preprocess_ai4i_data("data/raw/ai4i2020.csv", "data/processed")
    processed_data = pd.read_csv("data/processed/processed_data.csv")
    
    # Create extended features
    engineer = FeatureEngineer()
    extended_features = engineer.get_feature_set(processed_data, 'extended')
    
    # Separate features and target
    target_col = 'Machine failure'
    X = extended_features.drop(columns=[target_col])
    y = extended_features[target_col]
    
    # Create comparison feature sets
    feature_sets = create_comparison_feature_sets(X, y, "data/features")
    
    # Print results
    for name, features in feature_sets.items():
        print(f"{name}: {len(features)} features")
