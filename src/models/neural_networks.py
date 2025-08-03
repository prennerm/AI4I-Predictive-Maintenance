"""
Neural Networks Module

This module provides deep learning implementations optimized for predictive maintenance:
- Multi-Layer Perceptron (MLP) for tabular data
- Convolutional Neural Networks (CNN) for pattern recognition
- LSTM networks for time-series analysis
- Autoencoder for anomaly detection
- Deep Feature Learning networks

All models inherit from BaseModel classes and are fully compatible with
the ModelTrainer and ModelEvaluator frameworks.

Author: AI4I Project Team
Created: August 2025
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import pickle

# TensorFlow/Keras imports with fallback handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    TENSORFLOW_AVAILABLE = True
    
    # Configure TensorFlow
    tf.get_logger().setLevel('ERROR')  # Reduce TensorFlow logging
    
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")

# PyTorch imports (optional alternative)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch")

# Our base classes
from src.models.base_model import BaseModel, ClassificationModel, RegressionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPClassifier(ClassificationModel):
    """
    Multi-Layer Perceptron for classification tasks optimized for predictive maintenance.
    """
    
    def __init__(self,
                 hidden_layers: List[int] = [128, 64, 32],
                 activation: str = 'relu',
                 dropout_rate: float = 0.3,
                 l2_reg: float = 0.001,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 early_stopping_patience: int = 10,
                 validation_split: float = 0.2,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize MLP Classifier.
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function for hidden layers
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization strength
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            epochs: Maximum training epochs
            early_stopping_patience: Early stopping patience
            validation_split: Validation data split ratio
            random_state: Random state for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__()
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for neural networks. Install with: pip install tensorflow")
        
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.random_state = random_state
        
        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        self.input_dim = None
        self.n_classes = None
        
        # Hyperparameter space for tuning
        self.hyperparameter_space = {
            'hidden_layers': [
                [64, 32], [128, 64], [128, 64, 32], [256, 128, 64], [512, 256, 128]
            ],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
            'l2_reg': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': [0.0001, 0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64, 128]
        }
        
        logger.info("MLPClassifier initialized")
    
    def _build_model(self) -> keras.Model:
        """Build the neural network architecture."""
        
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            self.hidden_layers[0],
            input_dim=self.input_dim,
            activation=self.activation,
            kernel_regularizer=regularizers.l2(self.l2_reg)
        ))
        model.add(layers.Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ))
            model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        if self.n_classes == 2:
            # Binary classification
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            # Multi-class classification
            model.add(layers.Dense(self.n_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the MLP model.
        
        Args:
            X: Training features
            y: Training targets
        """
        logger.info("Training MLP Classifier...")
        
        # Store data characteristics
        self.input_dim = X.shape[1]
        self.n_classes = len(np.unique(y))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Prepare target for Keras
        if self.n_classes == 2:
            y_final = y_encoded.astype(np.float32)
        else:
            y_final = to_categorical(y_encoded, num_classes=self.n_classes)
        
        # Build model
        self.model = self._build_model()
        
        # Setup callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_scaled, y_final,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks_list,
            verbose=0
        )
        
        self.is_fitted = True
        logger.info(f"MLP training completed in {len(self.history.history['loss'])} epochs")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        if self.n_classes == 2:
            # Binary classification
            predictions = self.model.predict(X_scaled, verbose=0)
            return (predictions > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            predictions = self.model.predict(X_scaled, verbose=0)
            return self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict(X_scaled, verbose=0)
        
        if self.n_classes == 2:
            # Binary classification - return probabilities for both classes
            prob_class_1 = probabilities.flatten()
            prob_class_0 = 1 - prob_class_1
            return np.column_stack([prob_class_0, prob_class_1])
        else:
            # Multi-class classification
            return probabilities
    
    def get_hyperparameter_space(self) -> Dict[str, List[Any]]:
        """Get hyperparameter space for tuning."""
        return self.hyperparameter_space


class MLPRegressor(RegressionModel):
    """Multi-Layer Perceptron for regression tasks."""
    
    def __init__(self,
                 hidden_layers: List[int] = [128, 64, 32],
                 activation: str = 'relu',
                 dropout_rate: float = 0.3,
                 l2_reg: float = 0.001,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 early_stopping_patience: int = 10,
                 validation_split: float = 0.2,
                 random_state: int = 42,
                 **kwargs):
        """Initialize MLP Regressor."""
        super().__init__()
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for neural networks")
        
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.random_state = random_state
        
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.history = None
        self.input_dim = None
        
        self.hyperparameter_space = {
            'hidden_layers': [
                [64, 32], [128, 64], [128, 64, 32], [256, 128, 64]
            ],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4],
            'l2_reg': [0.0001, 0.001, 0.01],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [16, 32, 64]
        }
    
    def _build_model(self) -> keras.Model:
        """Build regression model."""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            self.hidden_layers[0],
            input_dim=self.input_dim,
            activation=self.activation,
            kernel_regularizer=regularizers.l2(self.l2_reg)
        ))
        model.add(layers.Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ))
            model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer (no activation for regression)
        model.add(layers.Dense(1))
        
        # Compile
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the regression model."""
        logger.info("Training MLP Regressor...")
        
        self.input_dim = X.shape[1]
        
        # Scale features and targets
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Build model
        self.model = self._build_model()
        
        # Setup callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_scaled, y_scaled,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks_list,
            verbose=0
        )
        
        self.is_fitted = True
        logger.info("MLP Regressor training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make regression predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions_scaled = self.model.predict(X_scaled, verbose=0)
        
        # Inverse transform predictions
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        return predictions.flatten()


class LSTMClassifier(ClassificationModel):
    """
    LSTM network for time-series classification in predictive maintenance.
    """
    
    def __init__(self,
                 lstm_units: List[int] = [64, 32],
                 dense_layers: List[int] = [32],
                 dropout_rate: float = 0.3,
                 recurrent_dropout: float = 0.2,
                 l2_reg: float = 0.001,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 sequence_length: int = 10,
                 early_stopping_patience: int = 15,
                 validation_split: float = 0.2,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize LSTM Classifier.
        
        Args:
            lstm_units: List of LSTM layer sizes
            dense_layers: List of dense layer sizes after LSTM
            dropout_rate: Dropout rate
            recurrent_dropout: Recurrent dropout rate
            l2_reg: L2 regularization
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Training epochs
            sequence_length: Input sequence length
            early_stopping_patience: Early stopping patience
            validation_split: Validation split ratio
            random_state: Random seed
        """
        super().__init__()
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM networks")
        
        self.lstm_units = lstm_units
        self.dense_layers = dense_layers
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.random_state = random_state
        
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        self.n_features = None
        self.n_classes = None
        
        self.hyperparameter_space = {
            'lstm_units': [[32], [64], [32, 16], [64, 32], [128, 64]],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'recurrent_dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.0001, 0.001, 0.01],
            'sequence_length': [5, 10, 15, 20]
        }
        
        logger.info("LSTMClassifier initialized")
    
    def _prepare_sequences(self, X: np.ndarray) -> np.ndarray:
        """Convert tabular data to sequences for LSTM."""
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Create sequences by sliding window
        sequences = []
        for i in range(self.sequence_length, n_samples):
            sequence = X[i-self.sequence_length:i]
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def _build_model(self) -> keras.Model:
        """Build LSTM model architecture."""
        model = models.Sequential()
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)  # Return sequences except for last layer
            
            if i == 0:
                # First LSTM layer with input shape
                model.add(layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout,
                    kernel_regularizer=regularizers.l2(self.l2_reg),
                    input_shape=(self.sequence_length, self.n_features)
                ))
            else:
                model.add(layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout,
                    kernel_regularizer=regularizers.l2(self.l2_reg)
                ))
        
        # Dense layers
        for units in self.dense_layers:
            model.add(layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ))
            model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        if self.n_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(self.n_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
        
        # Compile
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train LSTM model."""
        logger.info("Training LSTM Classifier...")
        
        # Store characteristics
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_sequences = self._prepare_sequences(X_scaled)
        y_sequences = y[self.sequence_length:]  # Align targets with sequences
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_sequences)
        
        if self.n_classes == 2:
            y_final = y_encoded.astype(np.float32)
        else:
            y_final = to_categorical(y_encoded, num_classes=self.n_classes)
        
        # Build model
        self.model = self._build_model()
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_sequences, y_final,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks_list,
            verbose=0
        )
        
        self.is_fitted = True
        logger.info("LSTM training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make LSTM predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        X_sequences = self._prepare_sequences(X_scaled)
        
        if self.n_classes == 2:
            predictions = self.model.predict(X_sequences, verbose=0)
            return (predictions > 0.5).astype(int).flatten()
        else:
            predictions = self.model.predict(X_sequences, verbose=0)
            return self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get LSTM prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        X_sequences = self._prepare_sequences(X_scaled)
        probabilities = self.model.predict(X_sequences, verbose=0)
        
        if self.n_classes == 2:
            prob_class_1 = probabilities.flatten()
            prob_class_0 = 1 - prob_class_1
            return np.column_stack([prob_class_0, prob_class_1])
        else:
            return probabilities


class AutoEncoder(BaseModel):
    """
    Autoencoder for anomaly detection in predictive maintenance.
    """
    
    def __init__(self,
                 encoding_layers: List[int] = [128, 64, 32],
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 l2_reg: float = 0.001,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 early_stopping_patience: int = 10,
                 validation_split: float = 0.2,
                 contamination_rate: float = 0.1,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Autoencoder.
        
        Args:
            encoding_layers: Encoder layer sizes (decoder mirrors these)
            activation: Activation function
            dropout_rate: Dropout rate
            l2_reg: L2 regularization
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Training epochs
            early_stopping_patience: Early stopping patience
            validation_split: Validation split
            contamination_rate: Expected outlier rate
            random_state: Random seed
        """
        super().__init__()
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for Autoencoder")
        
        self.encoding_layers = encoding_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.contamination_rate = contamination_rate
        self.random_state = random_state
        
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.encoder = None
        self.decoder = None
        self.scaler = StandardScaler()
        self.history = None
        self.threshold = None
        self.input_dim = None
        
        logger.info("AutoEncoder initialized")
    
    def _build_model(self) -> Tuple[keras.Model, keras.Model, keras.Model]:
        """Build autoencoder architecture."""
        
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = input_layer
        for units in self.encoding_layers:
            encoded = layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2_reg)
            )(encoded)
            encoded = layers.Dropout(self.dropout_rate)(encoded)
        
        # Decoder (mirror of encoder)
        decoded = encoded
        for units in reversed(self.encoding_layers[:-1]):
            decoded = layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2_reg)
            )(decoded)
            decoded = layers.Dropout(self.dropout_rate)(decoded)
        
        # Output layer (same size as input)
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # Create models
        autoencoder = models.Model(input_layer, decoded)
        encoder = models.Model(input_layer, encoded)
        
        # Compile autoencoder
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        autoencoder.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder, encoder
    
    def train(self, X: np.ndarray, y: np.ndarray = None) -> None:
        """
        Train autoencoder (unsupervised).
        
        Args:
            X: Training features
            y: Not used (unsupervised learning)
        """
        logger.info("Training AutoEncoder...")
        
        self.input_dim = X.shape[1]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model
        self.model, self.encoder = self._build_model()
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            )
        ]
        
        # Train (input = output for autoencoder)
        self.history = self.model.fit(
            X_scaled, X_scaled,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks_list,
            verbose=0
        )
        
        # Calculate reconstruction threshold
        reconstructions = self.model.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        self.threshold = np.percentile(reconstruction_errors, (1 - self.contamination_rate) * 100)
        
        self.is_fitted = True
        logger.info(f"AutoEncoder training completed. Anomaly threshold: {self.threshold:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (1 = anomaly, 0 = normal).
        
        Args:
            X: Input features
            
        Returns:
            Anomaly predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        reconstructions = self.model.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        return (reconstruction_errors > self.threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (reconstruction errors).
        
        Args:
            X: Input features
            
        Returns:
            Anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        reconstructions = self.model.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Normalize scores to [0,1] range
        normalized_scores = reconstruction_errors / (self.threshold * 2)
        normalized_scores = np.clip(normalized_scores, 0, 1)
        
        # Return as probability matrix
        prob_normal = 1 - normalized_scores
        prob_anomaly = normalized_scores
        
        return np.column_stack([prob_normal, prob_anomaly])
    
    def get_encoding(self, X: np.ndarray) -> np.ndarray:
        """Get encoded representations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before encoding")
        
        X_scaled = self.scaler.transform(X)
        return self.encoder.predict(X_scaled, verbose=0)


# CNN for pattern recognition (if 2D data available)
class CNNClassifier(ClassificationModel):
    """
    Convolutional Neural Network for pattern recognition in maintenance data.
    """
    
    def __init__(self,
                 conv_layers: List[Tuple[int, int]] = [(32, 3), (64, 3)],
                 dense_layers: List[int] = [128, 64],
                 dropout_rate: float = 0.3,
                 l2_reg: float = 0.001,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 early_stopping_patience: int = 10,
                 validation_split: float = 0.2,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize CNN Classifier.
        
        Args:
            conv_layers: List of (filters, kernel_size) tuples
            dense_layers: Dense layer sizes
            dropout_rate: Dropout rate
            l2_reg: L2 regularization
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Training epochs
            early_stopping_patience: Early stopping patience
            validation_split: Validation split
            random_state: Random seed
        """
        super().__init__()
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN")
        
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.random_state = random_state
        
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        self.input_shape = None
        self.n_classes = None
        
        logger.info("CNNClassifier initialized")
    
    def _reshape_for_cnn(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape tabular data for CNN.
        This is a simple approach - in practice, domain knowledge should guide reshaping.
        """
        n_samples, n_features = X.shape
        
        # Simple square-ish reshaping
        height = int(np.sqrt(n_features))
        width = n_features // height
        
        if height * width != n_features:
            # Pad if necessary
            pad_size = height * (width + 1) - n_features
            X_padded = np.pad(X, ((0, 0), (0, pad_size)), mode='constant')
            width += 1
        else:
            X_padded = X
        
        return X_padded.reshape(n_samples, height, width, 1)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train CNN model."""
        logger.info("Training CNN Classifier...")
        
        # Store characteristics
        self.n_classes = len(np.unique(y))
        
        # Scale and reshape
        X_scaled = self.scaler.fit_transform(X)
        X_reshaped = self._reshape_for_cnn(X_scaled)
        self.input_shape = X_reshaped.shape[1:]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        if self.n_classes == 2:
            y_final = y_encoded.astype(np.float32)
        else:
            y_final = to_categorical(y_encoded, num_classes=self.n_classes)
        
        # Build and compile model
        self.model = self._build_cnn_model()
        
        # Train
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            )
        ]
        
        self.history = self.model.fit(
            X_reshaped, y_final,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks_list,
            verbose=0
        )
        
        self.is_fitted = True
        logger.info("CNN training completed")
    
    def _build_cnn_model(self) -> keras.Model:
        """Build CNN architecture."""
        model = models.Sequential()
        
        # Convolutional layers
        for i, (filters, kernel_size) in enumerate(self.conv_layers):
            if i == 0:
                model.add(layers.Conv2D(
                    filters, kernel_size,
                    activation='relu',
                    input_shape=self.input_shape,
                    kernel_regularizer=regularizers.l2(self.l2_reg)
                ))
            else:
                model.add(layers.Conv2D(
                    filters, kernel_size,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(self.l2_reg)
                ))
            
            model.add(layers.MaxPooling2D(2))
            model.add(layers.Dropout(self.dropout_rate))
        
        # Flatten for dense layers
        model.add(layers.Flatten())
        
        # Dense layers
        for units in self.dense_layers:
            model.add(layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ))
            model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        if self.n_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(self.n_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
        
        # Compile
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make CNN predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        X_reshaped = self._reshape_for_cnn(X_scaled)
        
        if self.n_classes == 2:
            predictions = self.model.predict(X_reshaped, verbose=0)
            return (predictions > 0.5).astype(int).flatten()
        else:
            predictions = self.model.predict(X_reshaped, verbose=0)
            return self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))


# Neural Network Factory
class NeuralNetworkFactory:
    """Factory for creating neural network models."""
    
    AVAILABLE_MODELS = {
        'mlp_classifier': MLPClassifier,
        'mlp_regressor': MLPRegressor,
        'lstm_classifier': LSTMClassifier,
        'autoencoder': AutoEncoder,
        'cnn_classifier': CNNClassifier
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """Create a neural network model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for neural networks. Install with: pip install tensorflow")
        
        if model_type not in cls.AVAILABLE_MODELS:
            available = list(cls.AVAILABLE_MODELS.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
        
        model_class = cls.AVAILABLE_MODELS[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def create_model_suite(cls, problem_type: str = 'classification') -> List[BaseModel]:
        """Create a suite of neural network models."""
        models = []
        
        if problem_type == 'classification':
            # MLP Classifier with different architectures
            models.append(cls.create_model('mlp_classifier', hidden_layers=[64, 32]))
            models.append(cls.create_model('mlp_classifier', hidden_layers=[128, 64, 32]))
            
            # LSTM for sequential patterns
            models.append(cls.create_model('lstm_classifier', lstm_units=[32], sequence_length=10))
            
            # CNN for pattern recognition
            models.append(cls.create_model('cnn_classifier', conv_layers=[(16, 3), (32, 3)]))
            
        elif problem_type == 'regression':
            models.append(cls.create_model('mlp_regressor', hidden_layers=[64, 32]))
            models.append(cls.create_model('mlp_regressor', hidden_layers=[128, 64, 32]))
        
        elif problem_type == 'anomaly_detection':
            models.append(cls.create_model('autoencoder', encoding_layers=[64, 32, 16]))
            models.append(cls.create_model('autoencoder', encoding_layers=[128, 64, 32]))
        
        logger.info(f"Created neural network suite with {len(models)} models")
        return models
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available neural network models."""
        return list(cls.AVAILABLE_MODELS.keys())


# Convenience functions
def create_optimized_neural_networks(problem_type: str = 'classification') -> List[BaseModel]:
    """Create optimized neural networks for predictive maintenance."""
    factory = NeuralNetworkFactory()
    
    if problem_type == 'classification':
        models = [
            # Optimized MLP
            factory.create_model(
                'mlp_classifier',
                hidden_layers=[256, 128, 64],
                dropout_rate=0.3,
                l2_reg=0.001,
                learning_rate=0.001,
                epochs=150
            ),
            
            # Optimized LSTM
            factory.create_model(
                'lstm_classifier',
                lstm_units=[64, 32],
                dense_layers=[32],
                sequence_length=15,
                epochs=150
            )
        ]
    
    elif problem_type == 'regression':
        models = [
            factory.create_model(
                'mlp_regressor',
                hidden_layers=[256, 128, 64],
                dropout_rate=0.2,
                learning_rate=0.001,
                epochs=150
            )
        ]
    
    return models


# Testing function
def test_neural_networks():
    """Test neural network implementations."""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available, skipping neural network tests")
        return
    
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    
    print("Testing Neural Networks...")
    
    # Test MLP Classifier
    X_cls, y_cls = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    
    factory = NeuralNetworkFactory()
    
    # Test MLP
    mlp = factory.create_model('mlp_classifier', hidden_layers=[32, 16], epochs=10)
    mlp.train(X_train, y_train)
    predictions = mlp.predict(X_test)
    probabilities = mlp.predict_proba(X_test)
    
    assert len(predictions) == len(X_test)
    assert probabilities.shape == (len(X_test), 2)
    
    # Test Autoencoder
    autoencoder = factory.create_model('autoencoder', encoding_layers=[16, 8], epochs=10)
    autoencoder.train(X_train)  # Unsupervised
    anomaly_predictions = autoencoder.predict(X_test)
    
    assert len(anomaly_predictions) == len(X_test)
    
    print("Neural network tests passed!")


if __name__ == "__main__":
    test_neural_networks()
