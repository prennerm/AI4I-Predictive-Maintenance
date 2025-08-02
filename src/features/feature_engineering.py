"""
Feature Engineering Module for AI4I Predictive Maintenance Project

This module provides comprehensive feature engineering functionality including:
- Baseline feature preparation (original dataset features)
- Advanced feature creation (domain-specific engineering)
- Statistical feature derivation
- Interaction features between variables
- Feature set management for comparison experiments

Author: AI4I Team
Date: August 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Comprehensive feature engineering class for the AI4I dataset.
    
    This class handles creation of different feature sets:
    - Baseline: Original dataset features
    - Extended: Enhanced with domain-specific engineered features
    - Statistical: Additional statistical transformations
    """
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.baseline_features = None
        self.engineered_features = None
        self.feature_sets = {}
        
    def prepare_baseline_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare baseline features (original dataset features).
        
        Args:
            data (pd.DataFrame): Preprocessed dataset
            
        Returns:
            pd.DataFrame: Baseline feature set
        """
        logger.info("Preparing baseline features")
        
        # Define baseline features (original AI4I features)
        baseline_cols = [
            'Air temperature [K]',
            'Process temperature [K]', 
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ]
        
        # Add Product Type encoded features if they exist
        product_type_cols = [col for col in data.columns if col.startswith('Type_')]
        baseline_cols.extend(product_type_cols)
        
        # Select available columns
        available_cols = [col for col in baseline_cols if col in data.columns]
        self.baseline_features = data[available_cols].copy()
        
        logger.info(f"Baseline features prepared: {len(available_cols)} features")
        return self.baseline_features
    
    def create_temperature_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create temperature-related engineered features.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Temperature features
        """
        temp_features = pd.DataFrame(index=data.index)
        
        if 'Air temperature [K]' in data.columns and 'Process temperature [K]' in data.columns:
            # Temperature difference (process heat generation)
            temp_features['temp_difference'] = data['Process temperature [K]'] - data['Air temperature [K]']
            
            # Temperature ratio
            temp_features['temp_ratio'] = data['Process temperature [K]'] / data['Air temperature [K]']
            
            # Temperature efficiency indicator
            temp_features['temp_efficiency'] = temp_features['temp_difference'] / data['Air temperature [K]']
            
            # Convert Kelvin to Celsius for some calculations
            air_temp_c = data['Air temperature [K]'] - 273.15
            proc_temp_c = data['Process temperature [K]'] - 273.15
            
            # Temperature volatility (squared difference)
            temp_features['temp_volatility'] = (proc_temp_c - air_temp_c) ** 2
            
        logger.info("Temperature features created")
        return temp_features
    
    def create_power_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create power and energy-related features.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Power features
        """
        power_features = pd.DataFrame(index=data.index)
        
        if 'Rotational speed [rpm]' in data.columns and 'Torque [Nm]' in data.columns:
            # Convert RPM to rad/s for power calculation
            omega = data['Rotational speed [rpm]'] * 2 * np.pi / 60
            
            # Mechanical power (P = τ × ω)
            power_features['mechanical_power'] = data['Torque [Nm]'] * omega
            
            # Power density (power per unit torque)
            power_features['power_density'] = power_features['mechanical_power'] / (data['Torque [Nm]'] + 1e-6)
            
            # Specific power (power per RPM)
            power_features['specific_power'] = power_features['mechanical_power'] / (data['Rotational speed [rpm]'] + 1e-6)
            
            # Torque efficiency at given speed
            power_features['torque_efficiency'] = data['Torque [Nm]'] / (data['Rotational speed [rpm]'] + 1e-6)
            
        logger.info("Power features created")
        return power_features
    
    def create_wear_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create tool wear-related features.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Wear features
        """
        wear_features = pd.DataFrame(index=data.index)
        
        if 'Tool wear [min]' in data.columns:
            # Wear rate categories
            wear_features['wear_low'] = (data['Tool wear [min]'] < 100).astype(int)
            wear_features['wear_medium'] = ((data['Tool wear [min]'] >= 100) & 
                                          (data['Tool wear [min]'] < 200)).astype(int)
            wear_features['wear_high'] = (data['Tool wear [min]'] >= 200).astype(int)
            
            # Logarithmic wear (to handle non-linear wear patterns)
            wear_features['wear_log'] = np.log1p(data['Tool wear [min]'])
            
            # Wear squared (wear accelerates non-linearly)
            wear_features['wear_squared'] = data['Tool wear [min]'] ** 2
            
            # Wear interaction with other variables
            if 'Rotational speed [rpm]' in data.columns:
                wear_features['wear_speed_interaction'] = (data['Tool wear [min]'] * 
                                                         data['Rotational speed [rpm]'] / 1000)
            
            if 'Torque [Nm]' in data.columns:
                wear_features['wear_torque_interaction'] = (data['Tool wear [min]'] * 
                                                          data['Torque [Nm]'] / 100)
        
        logger.info("Wear features created")
        return wear_features
    
    def create_operational_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create operational efficiency and stress features.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Operational features
        """
        operational_features = pd.DataFrame(index=data.index)
        
        # Create operational stress indicators
        if all(col in data.columns for col in ['Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']):
            # Overall machine stress index
            speed_norm = data['Rotational speed [rpm]'] / data['Rotational speed [rpm]'].max()
            torque_norm = data['Torque [Nm]'] / data['Torque [Nm]'].max()
            wear_norm = data['Tool wear [min]'] / data['Tool wear [min]'].max()
            
            operational_features['stress_index'] = (speed_norm + torque_norm + wear_norm) / 3
            
            # Operational efficiency score
            operational_features['efficiency_score'] = (speed_norm * torque_norm) / (wear_norm + 0.1)
            
        # Speed categories
        if 'Rotational speed [rpm]' in data.columns:
            speed_quartiles = data['Rotational speed [rpm]'].quantile([0.25, 0.5, 0.75])
            operational_features['speed_low'] = (data['Rotational speed [rpm]'] <= speed_quartiles[0.25]).astype(int)
            operational_features['speed_medium'] = ((data['Rotational speed [rpm]'] > speed_quartiles[0.25]) & 
                                                   (data['Rotational speed [rpm]'] <= speed_quartiles[0.75])).astype(int)
            operational_features['speed_high'] = (data['Rotational speed [rpm]'] > speed_quartiles[0.75]).astype(int)
        
        # Torque categories
        if 'Torque [Nm]' in data.columns:
            torque_quartiles = data['Torque [Nm]'].quantile([0.25, 0.5, 0.75])
            operational_features['torque_low'] = (data['Torque [Nm]'] <= torque_quartiles[0.25]).astype(int)
            operational_features['torque_medium'] = ((data['Torque [Nm]'] > torque_quartiles[0.25]) & 
                                                    (data['Torque [Nm]'] <= torque_quartiles[0.75])).astype(int)
            operational_features['torque_high'] = (data['Torque [Nm]'] > torque_quartiles[0.75]).astype(int)
        
        logger.info("Operational features created")
        return operational_features
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Interaction features
        """
        interaction_features = pd.DataFrame(index=data.index)
        
        # Temperature and operational interactions
        if all(col in data.columns for col in ['Process temperature [K]', 'Rotational speed [rpm]']):
            interaction_features['temp_speed_interaction'] = (data['Process temperature [K]'] * 
                                                            data['Rotational speed [rpm]'] / 1000)
        
        if all(col in data.columns for col in ['Process temperature [K]', 'Torque [Nm]']):
            interaction_features['temp_torque_interaction'] = (data['Process temperature [K]'] * 
                                                             data['Torque [Nm]'] / 100)
        
        # Speed and torque relationship
        if all(col in data.columns for col in ['Rotational speed [rpm]', 'Torque [Nm]']):
            interaction_features['speed_torque_ratio'] = (data['Rotational speed [rpm]'] / 
                                                        (data['Torque [Nm]'] + 1e-6))
            interaction_features['speed_torque_product'] = (data['Rotational speed [rpm]'] * 
                                                          data['Torque [Nm]'] / 1000)
        
        logger.info("Interaction features created")
        return interaction_features
    
    def create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical transformation features.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Statistical features
        """
        statistical_features = pd.DataFrame(index=data.index)
        
        # Define numeric columns for statistical transformations
        numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 
                       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        available_numeric_cols = [col for col in numeric_cols if col in data.columns]
        
        for col in available_numeric_cols:
            # Squared terms (capture non-linear relationships)
            statistical_features[f'{col}_squared'] = data[col] ** 2
            
            # Cubic terms (for more complex relationships)
            statistical_features[f'{col}_cubed'] = data[col] ** 3
            
            # Logarithmic transformation (for exponential relationships)
            statistical_features[f'{col}_log'] = np.log1p(data[col])
            
            # Square root transformation
            statistical_features[f'{col}_sqrt'] = np.sqrt(data[col])
            
            # Standardized values (z-score within this feature)
            statistical_features[f'{col}_zscore'] = ((data[col] - data[col].mean()) / 
                                                    (data[col].std() + 1e-6))
        
        logger.info("Statistical features created")
        return statistical_features
    
    def create_extended_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create the complete set of engineered features.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: All engineered features combined
        """
        logger.info("Creating extended feature set")
        
        # Create all feature categories
        temp_features = self.create_temperature_features(data)
        power_features = self.create_power_features(data)
        wear_features = self.create_wear_features(data)
        operational_features = self.create_operational_features(data)
        interaction_features = self.create_interaction_features(data)
        statistical_features = self.create_statistical_features(data)
        
        # Combine all engineered features
        all_engineered = pd.concat([
            temp_features,
            power_features,
            wear_features,
            operational_features,
            interaction_features,
            statistical_features
        ], axis=1)
        
        # Handle any potential NaN values
        all_engineered = all_engineered.fillna(0)
        
        self.engineered_features = all_engineered
        logger.info(f"Extended features created: {all_engineered.shape[1]} features")
        
        return all_engineered
    
    def get_feature_set(self, data: pd.DataFrame, feature_set_type: str = 'baseline') -> pd.DataFrame:
        """
        Get specific feature set for model training.
        
        Args:
            data (pd.DataFrame): Input dataset
            feature_set_type (str): Type of feature set to return
                                  'baseline', 'extended', 'baseline_selected', 'extended_selected'
                                  
        Returns:
            pd.DataFrame: Requested feature set
        """
        if feature_set_type == 'baseline':
            return self.prepare_baseline_features(data)
            
        elif feature_set_type == 'extended':
            baseline = self.prepare_baseline_features(data)
            engineered = self.create_extended_features(data)
            return pd.concat([baseline, engineered], axis=1)
            
        elif feature_set_type == 'engineered_only':
            return self.create_extended_features(data)
            
        elif feature_set_type == 'temperature_focus':
            baseline = self.prepare_baseline_features(data)
            temp_features = self.create_temperature_features(data)
            power_features = self.create_power_features(data)
            return pd.concat([baseline, temp_features, power_features], axis=1)
            
        elif feature_set_type == 'wear_focus':
            baseline = self.prepare_baseline_features(data)
            wear_features = self.create_wear_features(data)
            operational_features = self.create_operational_features(data)
            return pd.concat([baseline, wear_features, operational_features], axis=1)
            
        else:
            logger.warning(f"Unknown feature set type: {feature_set_type}. Returning baseline.")
            return self.prepare_baseline_features(data)
    
    def get_available_feature_sets(self) -> List[str]:
        """
        Get list of available feature set types.
        
        Returns:
            List[str]: Available feature set types
        """
        return [
            'baseline',
            'extended', 
            'engineered_only',
            'temperature_focus',
            'wear_focus'
        ]
    
    def get_feature_info(self, feature_set_type: str = 'extended') -> Dict[str, int]:
        """
        Get information about features in a specific set.
        
        Args:
            feature_set_type (str): Type of feature set
            
        Returns:
            Dict[str, int]: Feature count information
        """
        # This is a placeholder - in practice, you'd create the features and count them
        feature_counts = {
            'baseline': 5,  # Original AI4I features
            'temperature': 5,  # Temperature-related features
            'power': 4,  # Power-related features  
            'wear': 6,  # Wear-related features
            'operational': 8,  # Operational features
            'interaction': 5,  # Interaction features
            'statistical': 25,  # Statistical transformations (5 features × 5 transforms)
        }
        
        if feature_set_type == 'baseline':
            return {'baseline_features': feature_counts['baseline']}
        elif feature_set_type == 'extended':
            total = sum(feature_counts.values())
            return {'total_features': total, **feature_counts}
        else:
            return feature_counts
    
    def save_feature_set(self, features: pd.DataFrame, output_dir: str, 
                        feature_set_name: str) -> None:
        """
        Save a feature set to disk.
        
        Args:
            features (pd.DataFrame): Feature set to save
            output_dir (str): Output directory
            feature_set_name (str): Name for the feature set
        """
        output_path = Path(output_dir) / f"{feature_set_name}_features.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        features.to_csv(output_path, index=False)
        logger.info(f"Feature set '{feature_set_name}' saved to: {output_path}")


def create_feature_sets_for_comparison(data: pd.DataFrame, output_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Create multiple feature sets for comparison experiments.
    
    Args:
        data (pd.DataFrame): Preprocessed dataset
        output_dir (str): Directory to save feature sets
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of feature sets
    """
    engineer = FeatureEngineer()
    
    feature_sets = {}
    
    # Create different feature sets
    for feature_type in engineer.get_available_feature_sets():
        feature_sets[feature_type] = engineer.get_feature_set(data, feature_type)
        engineer.save_feature_set(feature_sets[feature_type], output_dir, feature_type)
    
    logger.info(f"Created {len(feature_sets)} feature sets for comparison")
    
    return feature_sets


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from utils.data_preprocessing import preprocess_ai4i_data
    
    # Load and preprocess data
    summary = preprocess_ai4i_data("data/raw/ai4i2020.csv", "data/processed")
    
    # Load processed data
    processed_data = pd.read_csv("data/processed/processed_data.csv")
    
    # Create feature sets
    feature_sets = create_feature_sets_for_comparison(processed_data, "data/features")
    
    # Print feature set information
    engineer = FeatureEngineer()
    for feature_type in engineer.get_available_feature_sets():
        info = engineer.get_feature_info(feature_type)
        print(f"{feature_type}: {info}")
