"""
General helper functions for AI4I Predictive Maintenance Project

Contains utility functions used across different modules.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import pickle

def save_to_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save an object to a pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save the file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_from_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load an object from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_to_json(data: Dict, filepath: Union[str, Path]) -> None:
    """
    Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save the file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_from_json(filepath: Union[str, Path]) -> Dict:
    """
    Load a dictionary from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Target variable
        
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print comprehensive information about a DataFrame.
    
    Args:
        df: DataFrame to analyze
        name: Name to display
    """
    print(f"\n=== {name} Information ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

def get_feature_importance_summary(feature_names: List[str], 
                                 importances: np.ndarray,
                                 top_n: int = 10) -> pd.DataFrame:
    """
    Create a summary of feature importances.
    
    Args:
        feature_names: List of feature names
        importances: Array of importance values
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance summary
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    return importance_df.reset_index(drop=True)
