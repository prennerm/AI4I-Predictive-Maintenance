# AI4I 2020 Predictive Maintenance Project

## Project Overview

This project implements a production-ready machine learning pipeline for the AI4I 2020 Predictive Maintenance dataset. The architecture follows a **bottom-up, experiment-driven approach** focusing on technical excellence and reproducible ML workflows.

## Technical Objectives

- **Primary Goal**: Build a robust, experiment-driven ML pipeline for equipment failure prediction
- **Core Principles**: 
  - Production-ready Python scripts
  - Experiment versioning and comparison capabilities
  - Modular feature engineering with multiple strategy support
  - Comprehensive model comparison and visualization framework
  - Clean separation of concerns between data processing, modeling, and visualization

## Dataset Information

- **Source**: AI4I 2020 Predictive Maintenance Dataset
- **File**: `ai4i2020.csv` (10,000 samples)
- **Target**: Machine failure (main) + specific failure types (TWF, HDF, PWF, OSF, RNF)
- **Features**: 6 core predictors (Type, Air/Process temperature, Rotational speed, Torque, Tool wear)
- **Key Insight**: Identifier columns (UDI, Product ID) excluded to prevent data leakage

## Project Structure

```
AI4I/
├── README.md                          # Project documentation
├── architecture.md                   # This architecture document
├── data/
│   ├── raw/                          # Original dataset
│   ├── processed/                    # Cleaned and preprocessed data
│   └── features/                     # Engineered features
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Initial data analysis and experimentation
│   └── 02_experimentation.ipynb     # Model prototyping and testing (optional)
├── src/
│   ├── __init__.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py    # Feature creation and transformation
│   │   ├── feature_selection.py      # Feature selection algorithms
│   │   └── feature_utils.py          # Feature utility functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py             # Abstract base model class
│   │   ├── traditional_models.py     # RF, SVM, XGBoost implementations
│   │   ├── neural_networks.py        # Deep learning models
│   │   ├── model_trainer.py          # Training pipeline
│   │   ├── model_evaluator.py        # Evaluation metrics and validation
│   │   └── model_utils.py            # Model utilities and helpers
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── eda_plots.py              # Exploratory data analysis plots
│   │   ├── model_plots.py            # Model performance visualizations
│   │   └── report_plots.py           # Business report visualizations
│   └── utils/
│       ├── __init__.py
│       ├── config.py                 # Configuration management
│       ├── logger.py                 # Logging utilities
│       ├── data_preprocessing.py     # Comprehensive data preprocessing pipeline
│       └── helpers.py                # General helper functions
├── scripts/
│   ├── train_model.py                # Main training script
│   ├── evaluate_model.py             # Model evaluation script
│   ├── predict.py                    # Prediction script
│   └── generate_report.py            # Report generation script
├── models/                           # Saved model artifacts
├── reports/                          # Analysis reports and presentations
├── tests/                           # Unit tests
└── requirements.txt                  # Python dependencies
```

## Architecture Philosophy

This project implements a **lean, experiment-driven, bottom-up architecture**:

### Core Principles
- **Experiment-First Design**: All components support versioned experiments with different feature sets and model configurations
- **Script-Based Production**: Core functionality in production-ready Python scripts, notebooks only for initial exploration
- **Modular Feature Engineering**: Multiple feature engineering strategies that can be mixed and matched per experiment
- **Comprehensive Comparison Framework**: Built-in capabilities to compare models across different feature sets and experiments
- **Clean Data Practices**: Strict separation of features vs. targets, removal of data leakage sources (identifiers)

### Key Design Decisions
- **No Business Logic**: Focus purely on technical ML pipeline, no business insights or recommendations
- **Experiment Versioning**: Each experiment gets unique ID, tracked configurations, and isolated outputs
- **Feature Strategy Flexibility**: Support for different feature engineering approaches per experiment
- **Visualization-Driven Analysis**: Rich plotting capabilities for model comparison and feature analysis
- **Production Readiness**: All scripts designed for deployment and automation

## Methodology

### 1. Experiment-Driven Workflow
- Each experiment has unique configuration and isolated outputs
- Experiments can test different feature engineering strategies
- Same feature set can be tested across multiple models
- Same model can be tested across different feature sets
- All experiments are comparable and reproducible

### 2. Feature Engineering Framework
- **Baseline Strategy**: Minimal processing, original features only
- **Extended Strategy**: Domain-specific feature creation and transformations
- **Advanced Strategy**: Statistical derivations and complex feature interactions
- **Selection Integration**: Automatic feature selection within each strategy
- **Versioning**: Each feature set tagged and tracked per experiment

### 3. Model Training Pipeline
- **Multi-Algorithm Support**: Traditional ML (RF, SVM, XGBoost) and Deep Learning
- **Experiment Configuration**: Flexible configuration management for different setups  
- **Automated Evaluation**: Consistent metrics across all experiments
- **Model Persistence**: Organized storage of trained models per experiment
- **Performance Tracking**: Detailed logging and comparison capabilities

### 4. Visualization & Analysis
- **Experiment Comparison**: Side-by-side performance analysis
- **Feature Analysis**: Importance rankings and selection insights
- **Model Diagnostics**: Comprehensive performance visualizations
- **Cross-Experiment**: Compare same models on different features or vice versa

## Technical Performance Metrics

- **Model Performance**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Cross-Validation**: Stratified K-fold validation with confidence intervals
- **Feature Quality**: Feature importance rankings, selection stability
- **Experiment Tracking**: Performance across different feature sets and model configurations
- **Computational Efficiency**: Training time, inference speed, memory usage

## Technology Stack

- **Programming Language**: Python 3.8+
- **Data Processing**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn, xgboost, optuna (hyperparameter tuning)
- **Deep Learning**: pytorch/tensorflow (optional)
- **Visualization**: matplotlib, seaborn, plotly
- **Experiment Tracking**: Custom logging + file-based persistence
- **Development**: VS Code, Jupyter (exploration only)
- **Testing**: pytest, unittest

## Getting Started

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run Baseline Experiment
```bash
# Run experiment with minimal feature engineering
python scripts/train_model.py --experiment-id baseline_v1 --feature-strategy minimal

# Evaluate results
python scripts/evaluate_model.py --experiment-id baseline_v1
```

### 3. Feature Engineering Experiments
```bash
# Test different feature strategies
python scripts/train_model.py --experiment-id extended_v1 --feature-strategy extended
python scripts/train_model.py --experiment-id advanced_v1 --feature-strategy advanced

# Compare feature strategies
python scripts/evaluate_model.py --compare baseline_v1 extended_v1 advanced_v1
```

### 4. Model Comparison Experiments
```bash
# Same features, different models
python scripts/train_model.py --experiment-id baseline_rf --feature-strategy minimal --models random-forest
python scripts/train_model.py --experiment-id baseline_xgb --feature-strategy minimal --models xgboost

# Compare model performance
python scripts/evaluate_model.py --compare baseline_rf baseline_xgb
```

### 5. Generate Visualizations
```bash
# Generate experiment comparison plots
python scripts/generate_report.py --experiments baseline_v1 extended_v1 advanced_v1 --output reports/feature_comparison/
```

## Expected Deliverables

### Technical Outputs
1. **Production-Ready Scripts**: Complete ML pipeline as executable Python scripts
2. **Experiment Framework**: Versioned experiments with reproducible configurations
3. **Feature Engineering Library**: Multiple strategies for feature creation and selection
4. **Model Comparison System**: Comprehensive evaluation and visualization framework
5. **Performance Reports**: Technical analysis of model and feature performance

### Experiment Results
1. **Baseline Performance**: Results from minimal feature engineering approach
2. **Feature Strategy Comparison**: Analysis of different feature engineering approaches
3. **Model Performance Analysis**: Comparison across traditional ML and deep learning models
4. **Cross-Experiment Insights**: Technical findings from systematic experimentation

## Success Criteria

### Technical Excellence
- **Reproducibility**: All experiments fully reproducible with version control
- **Modularity**: Clean separation between feature engineering, modeling, and visualization
- **Performance**: Achieve >85% accuracy with proper cross-validation
- **Comparison Framework**: Ability to systematically compare models and feature strategies
- **Production Readiness**: Scripts ready for deployment and automation

### Experiment Framework
- **Multiple Feature Strategies**: Successfully implement and compare different approaches
- **Model Flexibility**: Support for various ML algorithms with consistent evaluation
- **Visualization Quality**: Comprehensive plots for model and experiment analysis
- **Code Quality**: Clean, tested, and maintainable codebase

---

*This project prioritizes technical excellence and systematic experimentation over business consulting, creating a robust foundation for predictive maintenance ML applications.*
