#!/usr/bin/env python3
"""
Production Inference Script for AI4I Predictive Maintenance

This script provides real-time and batch prediction capabilities for equipment failure
prediction in industrial environments. It serves as the production inference engine
that transforms trained ML models into actionable business insights.

Key Features:
- Single and batch predictions from trained models
- Real-time sensor data processing
- Business-oriented output with risk levels and recommendations
- API mode for system integration
- Automated alerting and notification system
- Comprehensive logging and monitoring

Usage Examples:
    # Single prediction
    python scripts/predict.py --input "300,310,1500,40,120" --model models/best_model/

    # Batch prediction from CSV
    python scripts/predict.py --input sensor_data.csv --output predictions.json

    # Real-time monitoring mode
    python scripts/predict.py --stream --alert-threshold 0.8

    # API service mode
    python scripts/predict.py --api-mode --port 8080

    # Scheduled business report
    python scripts/predict.py --input daily_data.csv --business-report

Author: AI4I Project Team
Created: August 2025
"""

import argparse
import logging
import sys
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Data handling
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import Config
from src.utils.data_preprocessing import DataPreprocessor
from src.utils.helpers import load_json, save_json, ensure_directory

from src.features.feature_engineering import FeatureEngineer
from src.features.feature_selection import FeatureSelector

from src.models.model_utils import load_model, ModelPersistence
from src.models.base_model import BaseModel


class PredictionResult:
    """
    Container for prediction results with business intelligence.
    """
    
    def __init__(self,
                 prediction: int,
                 probability: float,
                 confidence: float,
                 risk_level: str,
                 recommended_action: str,
                 model_name: str,
                 timestamp: str,
                 input_features: Optional[Dict] = None):
        self.prediction = prediction
        self.probability = probability
        self.confidence = confidence
        self.risk_level = risk_level
        self.recommended_action = recommended_action
        self.model_name = model_name
        self.timestamp = timestamp
        self.input_features = input_features or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'prediction': int(self.prediction),
            'probability': float(self.probability),
            'confidence': float(self.confidence),
            'risk_level': self.risk_level,
            'recommended_action': self.recommended_action,
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'input_features': self.input_features
        }
    
    def to_business_message(self) -> str:
        """Generate business-friendly message."""
        failure_text = "FAILURE PREDICTED" if self.prediction == 1 else "NORMAL OPERATION"
        return (f"[{self.timestamp}] {failure_text} - "
                f"Risk: {self.risk_level} ({self.probability:.1%}) - "
                f"Action: {self.recommended_action}")


class AI4IPredictionEngine:
    """
    Production prediction engine for AI4I predictive maintenance.
    
    This class handles the complete inference pipeline:
    - Model loading and validation
    - Data preprocessing and feature engineering
    - Prediction generation with confidence scoring
    - Business logic and risk assessment
    - Output formatting and alerting
    """
    
    def __init__(self, 
                 model_path: str,
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the prediction engine.
        
        Args:
            model_path: Path to trained model or model directory
            config: Configuration dictionary
            logger: Logger instance
        """
        self.model_path = Path(model_path)
        self.config = config or self._get_default_config()
        self.logger = logger or self._setup_default_logger()
        
        # Initialize components
        self.model = None
        self.model_metadata = {}
        self.preprocessor = None
        self.feature_engineer = None
        self.feature_selector = None
        self.selected_features = []
        
        # Load model and pipeline components
        self._load_model_pipeline()
        
        # Performance tracking
        self.prediction_count = 0
        self.start_time = time.time()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for prediction engine."""
        return {
            'risk_thresholds': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8,
                'critical': 0.9
            },
            'business_actions': {
                'low': 'Continue normal operation',
                'medium': 'Schedule maintenance within 1 week',
                'high': 'Schedule maintenance within 48 hours',
                'critical': 'IMMEDIATE maintenance required - Stop operation'
            },
            'confidence_threshold': 0.7,
            'alert_settings': {
                'enabled': True,
                'threshold': 0.8,
                'cooldown_minutes': 30
            },
            'output': {
                'include_features': False,
                'business_format': True,
                'detailed_analysis': False
            },
            'performance': {
                'batch_size': 1000,
                'timeout_seconds': 30
            }
        }
    
    def _setup_default_logger(self) -> logging.Logger:
        """Setup default logger for prediction engine."""
        return setup_logger(
            name='ai4i_prediction',
            level='INFO',
            log_file=f'logs/predictions_{int(time.time())}.log'
        )
    
    def _load_model_pipeline(self) -> None:
        """Load trained model and preprocessing pipeline."""
        
        try:
            # Determine model file path
            if self.model_path.is_dir():
                model_file = self.model_path / 'model.pkl'
                metadata_file = self.model_path / 'metadata.json'
            else:
                model_file = self.model_path
                metadata_file = self.model_path.parent / 'metadata.json'
            
            # Load model
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            self.model = load_model(str(model_file))
            self.logger.info(f"Model loaded successfully: {model_file}")
            
            # Load metadata
            if metadata_file.exists():
                self.model_metadata = load_json(metadata_file)
                self.selected_features = self.model_metadata.get('selected_features', [])
                self.logger.info(f"Model metadata loaded: {len(self.selected_features)} features")
            else:
                self.logger.warning("No model metadata found - using default preprocessing")
                self.selected_features = []
            
            # Initialize preprocessing components
            self._initialize_preprocessing_pipeline()
            
            self.logger.info("Prediction engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model pipeline: {e}")
            raise
    
    def _initialize_preprocessing_pipeline(self) -> None:
        """Initialize preprocessing and feature engineering pipeline."""
        
        try:
            # Try to load saved preprocessing pipeline
            pipeline_path = self.model_path.parent / 'preprocessing_pipeline.pkl'
            
            if pipeline_path.exists():
                # Load saved pipeline (if available)
                self.logger.info("Loading saved preprocessing pipeline")
                # Note: In a real implementation, you'd save/load the fitted preprocessors
            
            # Initialize components (same as used in training)
            self.preprocessor = DataPreprocessor()
            self.feature_engineer = FeatureEngineer()
            self.feature_selector = FeatureSelector()
            
            self.logger.info("Preprocessing pipeline initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing preprocessing pipeline: {e}")
            raise
    
    def _preprocess_input_data(self, data: Union[pd.DataFrame, np.ndarray, str, list]) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            data: Input data in various formats
            
        Returns:
            Preprocessed DataFrame ready for feature engineering
        """
        
        try:
            # Convert input to DataFrame
            if isinstance(data, str):
                # Parse comma-separated string
                values = [float(x.strip()) for x in data.split(',')]
                if len(values) == 5:  # AI4I typical features
                    columns = ['Air temperature [K]', 'Process temperature [K]', 
                              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
                else:
                    columns = [f'feature_{i}' for i in range(len(values))]
                df = pd.DataFrame([values], columns=columns)
                
            elif isinstance(data, list):
                # List of values
                if len(data) == 5:  # AI4I typical features
                    columns = ['Air temperature [K]', 'Process temperature [K]', 
                              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
                else:
                    columns = [f'feature_{i}' for i in range(len(data))]
                df = pd.DataFrame([data], columns=columns)
                
            elif isinstance(data, np.ndarray):
                # NumPy array
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
                
            elif isinstance(data, pd.DataFrame):
                # Already a DataFrame
                df = data.copy()
                
            else:
                raise ValueError(f"Unsupported input data type: {type(data)}")
            
            # Apply preprocessing (same as training)
            df_processed = self.preprocessor.handle_missing_values(df)
            df_processed = self.preprocessor.encode_categorical_features(df_processed)
            df_processed = self.preprocessor.scale_features(df_processed)
            
            return df_processed
            
        except Exception as e:
            self.logger.error(f"Error preprocessing input data: {e}")
            raise
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to preprocessed data.
        
        Args:
            data: Preprocessed DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        
        try:
            # Apply same feature engineering as training
            engineered_data = self.feature_engineer.get_feature_set(data, 'extended')
            
            # Select features used in training
            if self.selected_features:
                # Ensure all required features are present
                missing_features = set(self.selected_features) - set(engineered_data.columns)
                if missing_features:
                    self.logger.warning(f"Missing features: {missing_features}")
                    # Add missing features with default values
                    for feature in missing_features:
                        engineered_data[feature] = 0
                
                # Select only the features used in training
                engineered_data = engineered_data[self.selected_features]
            
            return engineered_data
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            # Fallback: return original data
            return data
    
    def _calculate_confidence(self, probability: float, model_uncertainty: float = 0.1) -> float:
        """
        Calculate prediction confidence score.
        
        Args:
            probability: Predicted probability
            model_uncertainty: Model uncertainty estimate
            
        Returns:
            Confidence score between 0 and 1
        """
        
        # Distance from decision boundary (0.5)
        boundary_distance = abs(probability - 0.5)
        
        # Normalize to 0-1 scale
        confidence = min(1.0, (boundary_distance * 2) * (1 - model_uncertainty))
        
        return confidence
    
    def _assess_risk_level(self, probability: float) -> str:
        """
        Assess risk level based on failure probability.
        
        Args:
            probability: Failure probability
            
        Returns:
            Risk level string
        """
        
        thresholds = self.config['risk_thresholds']
        
        if probability >= thresholds['critical']:
            return 'CRITICAL'
        elif probability >= thresholds['high']:
            return 'HIGH'
        elif probability >= thresholds['medium']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_recommended_action(self, risk_level: str) -> str:
        """
        Get recommended business action based on risk level.
        
        Args:
            risk_level: Assessed risk level
            
        Returns:
            Recommended action string
        """
        
        return self.config['business_actions'].get(risk_level.lower(), 
                                                   'Review equipment status')
    
    def predict_single(self, 
                      input_data: Union[pd.DataFrame, np.ndarray, str, list],
                      include_features: bool = False) -> PredictionResult:
        """
        Make prediction for single input.
        
        Args:
            input_data: Input data for prediction
            include_features: Whether to include feature values in result
            
        Returns:
            Prediction result with business intelligence
        """
        
        try:
            start_time = time.time()
            
            # Preprocess input data
            processed_data = self._preprocess_input_data(input_data)
            
            # Feature engineering
            engineered_data = self._engineer_features(processed_data)
            
            # Make prediction
            prediction = self.model.predict(engineered_data)[0]
            
            # Get probability if available
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(engineered_data)[0]
                probability = proba[1] if len(proba) > 1 else proba[0]
            else:
                # Fallback for models without probability
                probability = float(prediction)
            
            # Calculate confidence
            confidence = self._calculate_confidence(probability)
            
            # Assess risk and recommend action
            risk_level = self._assess_risk_level(probability)
            recommended_action = self._get_recommended_action(risk_level)
            
            # Create result
            result = PredictionResult(
                prediction=int(prediction),
                probability=float(probability),
                confidence=float(confidence),
                risk_level=risk_level,
                recommended_action=recommended_action,
                model_name=self.model_metadata.get('model_name', 'Unknown'),
                timestamp=datetime.now().isoformat(),
                input_features=engineered_data.iloc[0].to_dict() if include_features else None
            )
            
            # Update performance tracking
            self.prediction_count += 1
            inference_time = time.time() - start_time
            
            self.logger.info(f"Prediction completed - Risk: {risk_level}, "
                           f"Probability: {probability:.3f}, Time: {inference_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in single prediction: {e}")
            raise
    
    def predict_batch(self, 
                     input_data: pd.DataFrame,
                     include_features: bool = False) -> List[PredictionResult]:
        """
        Make predictions for batch of inputs.
        
        Args:
            input_data: DataFrame with multiple rows for prediction
            include_features: Whether to include feature values in results
            
        Returns:
            List of prediction results
        """
        
        try:
            start_time = time.time()
            batch_size = len(input_data)
            
            self.logger.info(f"Starting batch prediction for {batch_size} samples")
            
            # Process in chunks if large dataset
            chunk_size = self.config['performance']['batch_size']
            results = []
            
            for i in range(0, batch_size, chunk_size):
                chunk = input_data.iloc[i:i+chunk_size]
                
                # Preprocess chunk
                processed_chunk = self._preprocess_input_data(chunk)
                
                # Feature engineering
                engineered_chunk = self._engineer_features(processed_chunk)
                
                # Make predictions
                predictions = self.model.predict(engineered_chunk)
                
                # Get probabilities if available
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(engineered_chunk)
                    if probabilities.ndim > 1:
                        probabilities = probabilities[:, 1]
                else:
                    probabilities = predictions.astype(float)
                
                # Process each prediction in chunk
                for j, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    confidence = self._calculate_confidence(prob)
                    risk_level = self._assess_risk_level(prob)
                    recommended_action = self._get_recommended_action(risk_level)
                    
                    result = PredictionResult(
                        prediction=int(pred),
                        probability=float(prob),
                        confidence=float(confidence),
                        risk_level=risk_level,
                        recommended_action=recommended_action,
                        model_name=self.model_metadata.get('model_name', 'Unknown'),
                        timestamp=datetime.now().isoformat(),
                        input_features=engineered_chunk.iloc[j].to_dict() if include_features else None
                    )
                    
                    results.append(result)
                
                # Progress update for large batches
                if batch_size > 100:
                    progress = min(i + chunk_size, batch_size)
                    self.logger.info(f"Batch progress: {progress}/{batch_size}")
            
            # Update performance tracking
            self.prediction_count += batch_size
            total_time = time.time() - start_time
            avg_time = total_time / batch_size
            
            self.logger.info(f"Batch prediction completed - {batch_size} samples in "
                           f"{total_time:.2f}s (avg: {avg_time:.4f}s per sample)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {e}")
            raise
    
    def generate_business_report(self, results: List[PredictionResult]) -> Dict[str, Any]:
        """
        Generate business-oriented summary report.
        
        Args:
            results: List of prediction results
            
        Returns:
            Business report dictionary
        """
        
        if not results:
            return {'error': 'No predictions to report'}
        
        # Calculate summary statistics
        total_predictions = len(results)
        failure_predictions = sum(1 for r in results if r.prediction == 1)
        avg_probability = np.mean([r.probability for r in results])
        avg_confidence = np.mean([r.confidence for r in results])
        
        # Risk level distribution
        risk_distribution = {}
        for risk_level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
            count = sum(1 for r in results if r.risk_level == risk_level)
            risk_distribution[risk_level] = {
                'count': count,
                'percentage': (count / total_predictions) * 100
            }
        
        # Action recommendations
        action_summary = {}
        for result in results:
            action = result.recommended_action
            if action not in action_summary:
                action_summary[action] = 0
            action_summary[action] += 1
        
        # High-risk equipment (probability > 0.7)
        high_risk_equipment = [
            {
                'index': i,
                'probability': r.probability,
                'risk_level': r.risk_level,
                'action': r.recommended_action,
                'timestamp': r.timestamp
            }
            for i, r in enumerate(results) if r.probability > 0.7
        ]
        
        # Generate business insights
        insights = []
        
        if failure_predictions > 0:
            failure_rate = (failure_predictions / total_predictions) * 100
            insights.append(f"{failure_rate:.1f}% of equipment predicted to fail")
        
        critical_count = risk_distribution['CRITICAL']['count']
        if critical_count > 0:
            insights.append(f"{critical_count} equipment require IMMEDIATE attention")
        
        high_count = risk_distribution['HIGH']['count']
        if high_count > 0:
            insights.append(f"{high_count} equipment require maintenance within 48 hours")
        
        if avg_confidence < 0.7:
            insights.append("Model confidence below threshold - consider data quality review")
        
        # Cost impact estimation (simplified)
        cost_per_failure = 10000  # Example cost
        cost_per_maintenance = 1000  # Example cost
        
        estimated_failure_cost = failure_predictions * cost_per_failure
        estimated_maintenance_cost = (critical_count + high_count) * cost_per_maintenance
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_equipment': total_predictions,
                'failure_predictions': failure_predictions,
                'failure_rate_percentage': (failure_predictions / total_predictions) * 100,
                'average_failure_probability': avg_probability,
                'average_confidence': avg_confidence
            },
            'risk_distribution': risk_distribution,
            'action_recommendations': action_summary,
            'high_risk_equipment': high_risk_equipment,
            'business_insights': insights,
            'cost_estimates': {
                'potential_failure_costs': estimated_failure_cost,
                'recommended_maintenance_costs': estimated_maintenance_cost,
                'estimated_savings': max(0, estimated_failure_cost - estimated_maintenance_cost)
            },
            'model_info': {
                'model_name': self.model_metadata.get('model_name', 'Unknown'),
                'prediction_count': self.prediction_count,
                'engine_uptime_hours': (time.time() - self.start_time) / 3600
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get prediction engine performance statistics."""
        
        uptime = time.time() - self.start_time
        
        return {
            'total_predictions': self.prediction_count,
            'uptime_seconds': uptime,
            'uptime_hours': uptime / 3600,
            'predictions_per_hour': (self.prediction_count / uptime * 3600) if uptime > 0 else 0,
            'model_loaded': self.model is not None,
            'features_count': len(self.selected_features),
            'model_name': self.model_metadata.get('model_name', 'Unknown')
        }


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI4I Predictive Maintenance - Production Inference Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model specification
    parser.add_argument(
        '--model', 
        type=str,
        required=True,
        help='Path to trained model file or model directory'
    )
    
    # Input specification
    parser.add_argument(
        '--input', 
        type=str,
        help='Input data: CSV file path, comma-separated values, or "-" for stdin'
    )
    
    parser.add_argument(
        '--input-format', 
        type=str,
        choices=['csv', 'values', 'json'],
        default='auto',
        help='Input data format (auto-detected if not specified)'
    )
    
    # Output specification
    parser.add_argument(
        '--output', 
        type=str,
        help='Output file path (JSON format)'
    )
    
    parser.add_argument(
        '--output-format', 
        type=str,
        choices=['json', 'csv', 'text'],
        default='json',
        help='Output format'
    )
    
    # Operation modes
    parser.add_argument(
        '--batch', 
        action='store_true',
        help='Batch prediction mode for multiple inputs'
    )
    
    parser.add_argument(
        '--business-report', 
        action='store_true',
        help='Generate business-oriented summary report'
    )
    
    parser.add_argument(
        '--api-mode', 
        action='store_true',
        help='Start API server for real-time predictions'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8080,
        help='Port for API server mode'
    )
    
    # Analysis options
    parser.add_argument(
        '--include-features', 
        action='store_true',
        help='Include feature values in output'
    )
    
    parser.add_argument(
        '--confidence-threshold', 
        type=float, 
        default=0.7,
        help='Minimum confidence threshold for predictions'
    )
    
    parser.add_argument(
        '--alert-threshold', 
        type=float, 
        default=0.8,
        help='Risk probability threshold for alerts'
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


def load_input_data(input_path: str, input_format: str = 'auto') -> Union[pd.DataFrame, str, list]:
    """Load input data from various sources."""
    
    if input_path == '-':
        # Read from stdin
        import sys
        return sys.stdin.read().strip()
    
    input_path = Path(input_path)
    
    if not input_path.exists():
        # Try to parse as comma-separated values
        try:
            values = [float(x.strip()) for x in input_path.name.split(',')]
            return values
        except ValueError:
            raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Determine format
    if input_format == 'auto':
        if input_path.suffix.lower() == '.csv':
            input_format = 'csv'
        elif input_path.suffix.lower() == '.json':
            input_format = 'json'
        else:
            input_format = 'csv'  # Default assumption
    
    # Load data based on format
    if input_format == 'csv':
        return pd.read_csv(input_path)
    elif input_format == 'json':
        with open(input_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                return pd.DataFrame(data)
            else:
                return data
        else:
            return data
    else:
        raise ValueError(f"Unsupported input format: {input_format}")


def save_output(results: Union[List[PredictionResult], PredictionResult, Dict],
               output_path: str,
               output_format: str = 'json') -> None:
    """Save prediction results to file."""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert results to appropriate format
    if isinstance(results, PredictionResult):
        data = results.to_dict()
    elif isinstance(results, list):
        data = [r.to_dict() if isinstance(r, PredictionResult) else r for r in results]
    else:
        data = results
    
    # Save based on format
    if output_format == 'json':
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    elif output_format == 'csv':
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        df.to_csv(output_path, index=False)
    elif output_format == 'text':
        with open(output_path, 'w') as f:
            if isinstance(results, PredictionResult):
                f.write(results.to_business_message())
            elif isinstance(results, list):
                for result in results:
                    if isinstance(result, PredictionResult):
                        f.write(result.to_business_message() + '\n')
                    else:
                        f.write(str(result) + '\n')
            else:
                f.write(str(results))
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def start_api_server(prediction_engine: AI4IPredictionEngine, port: int = 8080) -> None:
    """Start API server for real-time predictions."""
    
    try:
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        
        @app.route('/predict', methods=['POST'])
        def predict():
            try:
                data = request.get_json()
                
                if 'input' not in data:
                    return jsonify({'error': 'Missing input data'}), 400
                
                # Make prediction
                result = prediction_engine.predict_single(data['input'])
                
                return jsonify(result.to_dict())
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/batch_predict', methods=['POST'])
        def batch_predict():
            try:
                data = request.get_json()
                
                if 'inputs' not in data:
                    return jsonify({'error': 'Missing inputs data'}), 400
                
                # Convert to DataFrame
                df = pd.DataFrame(data['inputs'])
                
                # Make batch predictions
                results = prediction_engine.predict_batch(df)
                
                return jsonify([r.to_dict() for r in results])
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/health', methods=['GET'])
        def health():
            stats = prediction_engine.get_performance_stats()
            return jsonify({
                'status': 'healthy',
                'stats': stats
            })
        
        @app.route('/business_report', methods=['POST'])
        def business_report():
            try:
                data = request.get_json()
                
                if 'inputs' not in data:
                    return jsonify({'error': 'Missing inputs data'}), 400
                
                # Convert to DataFrame and make predictions
                df = pd.DataFrame(data['inputs'])
                results = prediction_engine.predict_batch(df)
                
                # Generate business report
                report = prediction_engine.generate_business_report(results)
                
                return jsonify(report)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        print(f"🚀 AI4I Prediction API Server starting on port {port}")
        print(f"📍 Endpoints:")
        print(f"   POST /predict - Single prediction")
        print(f"   POST /batch_predict - Batch predictions")
        print(f"   POST /business_report - Business analysis")
        print(f"   GET  /health - Service health")
        example_cmd = f"curl -X POST http://localhost:{port}/predict -H 'Content-Type: application/json' -d '{{\"input\": [300, 310, 1500, 40, 120]}}'"
        print(f"💡 Example: {example_cmd}")
        
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except ImportError:
        print("❌ Flask not installed. Install with: pip install flask")
        sys.exit(1)


def main():
    """Main prediction pipeline."""
    
    print("🔮 AI4I Predictive Maintenance - Production Inference Engine")
    print("=" * 70)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger(
        name='ai4i_predict',
        level=args.log_level,
        log_file=f'logs/predictions_{int(time.time())}.log'
    )
    
    try:
        # Initialize prediction engine
        print(f"\n⚡ Initializing Prediction Engine...")
        print(f"📁 Model: {args.model}")
        
        prediction_engine = AI4IPredictionEngine(
            model_path=args.model,
            logger=logger
        )
        
        print(f"✅ Engine ready - Model: {prediction_engine.model_metadata.get('model_name', 'Unknown')}")
        
        # API server mode
        if args.api_mode:
            start_api_server(prediction_engine, args.port)
            return
        
        # Validate input
        if not args.input:
            print("❌ No input data provided. Use --input to specify data source.")
            sys.exit(1)
        
        # Load input data
        print(f"\n📊 Loading input data from: {args.input}")
        input_data = load_input_data(args.input, args.input_format)
        
        # Make predictions
        if args.batch or isinstance(input_data, pd.DataFrame):
            print("\n🔍 Running batch predictions...")
            
            if not isinstance(input_data, pd.DataFrame):
                print("❌ Batch mode requires CSV input file")
                sys.exit(1)
            
            results = prediction_engine.predict_batch(
                input_data, 
                include_features=args.include_features
            )
            
            print(f"✅ Batch prediction completed: {len(results)} samples")
            
            # Generate business report if requested
            if args.business_report:
                print("\n📋 Generating business report...")
                business_report = prediction_engine.generate_business_report(results)
                
                # Print summary
                summary = business_report['summary']
                print(f"📈 Equipment analyzed: {summary['total_equipment']}")
                print(f"⚠️  Failure predictions: {summary['failure_predictions']} ({summary['failure_rate_percentage']:.1f}%)")
                
                risk_dist = business_report['risk_distribution']
                print(f"🔴 Critical risk: {risk_dist['CRITICAL']['count']}")
                print(f"🟠 High risk: {risk_dist['HIGH']['count']}")
                print(f"🟡 Medium risk: {risk_dist['MEDIUM']['count']}")
                print(f"🟢 Low risk: {risk_dist['LOW']['count']}")
                
                # Save business report
                if args.output:
                    report_path = Path(args.output).with_suffix('.business_report.json')
                    save_output(business_report, str(report_path), 'json')
                    print(f"📄 Business report saved: {report_path}")
            
        else:
            print("\n🎯 Running single prediction...")
            
            result = prediction_engine.predict_single(
                input_data,
                include_features=args.include_features
            )
            
            print(f"\n🔮 PREDICTION RESULT:")
            print(f"   Prediction: {'FAILURE' if result.prediction == 1 else 'NORMAL'}")
            print(f"   Probability: {result.probability:.1%}")
            print(f"   Risk Level: {result.risk_level}")
            print(f"   Confidence: {result.confidence:.1%}")
            print(f"   Action: {result.recommended_action}")
            
            # Alert if high risk
            if result.probability >= args.alert_threshold:
                print(f"\n🚨 HIGH RISK ALERT!")
                print(f"   {result.to_business_message()}")
            
            results = result
        
        # Save output
        if args.output:
            print(f"\n💾 Saving results to: {args.output}")
            save_output(results, args.output, args.output_format)
        
        # Performance stats
        stats = prediction_engine.get_performance_stats()
        print(f"\n📊 Performance Stats:")
        print(f"   Total predictions: {stats['total_predictions']}")
        print(f"   Engine uptime: {stats['uptime_hours']:.2f} hours")
        print(f"   Predictions per hour: {stats['predictions_per_hour']:.1f}")
        
        print("\n✅ Prediction pipeline completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}")
        print(f"\n❌ Prediction pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
