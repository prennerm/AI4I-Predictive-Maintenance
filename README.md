# AI4I 2020 Predictive Maintenance

> Machine learning pipeline for equipment failure prediction using the AI4I 2020 dataset

## Overview

This project implements a production-ready, experiment-driven machine learning pipeline for predictive maintenance. The architecture prioritizes technical excellence, reproducible experiments, and systematic model comparison over business consulting.

**Key Features:**
- **Experiment-driven workflow** with versioned configurations
- **Multiple feature engineering strategies** (minimal, extended, advanced)
- **Comprehensive model comparison** across different algorithms and feature sets
- **Production-ready Python scripts** with CLI interfaces
- **Systematic evaluation framework** with cross-experiment analysis

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/prennerm/AI4I-Predictive-Maintenance.git
cd AI4I-Predictive-Maintenance

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Baseline Experiment

```bash
# Train models with minimal feature engineering
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

### 5. Generate Technical Reports

```bash
# Generate experiment comparison plots
python scripts/generate_report.py --experiments baseline_v1 extended_v1 advanced_v1 --output reports/feature_comparison/

# Generate comprehensive evaluation report
python scripts/generate_report.py --experiments baseline_v1 --technical-report
```

## Dataset Information

**AI4I 2020 Predictive Maintenance Dataset**
- **Samples**: 10,000 data points
- **Target**: Machine failure (binary classification)
- **Features**: 6 core predictors after data leakage prevention
  - Type (categorical)
  - Air temperature [K]
  - Process temperature [K] 
  - Rotational speed [rpm]
  - Torque [Nm]
  - Tool wear [min]
- **Key Insight**: UDI and Product ID excluded to prevent data leakage

## Experiment Framework

### Feature Engineering Strategies

1. **Minimal Strategy** (`--feature-strategy minimal`)
   - Uses only the 6 core features with basic preprocessing
   - Label encoding for categorical variables
   - Standard scaling for numerical features

2. **Extended Strategy** (`--feature-strategy extended`)
   - Includes domain-specific feature engineering
   - Temperature ratios and differences
   - Power calculations and efficiency metrics

3. **Advanced Strategy** (`--feature-strategy advanced`)
   - Complex feature interactions
   - Statistical derivations
   - Advanced temporal features

### Model Algorithms

Available models:
- `random-forest`: Random Forest Classifier
- `xgboost`: XGBoost Classifier  
- `logistic-regression`: Logistic Regression
- `svm`: Support Vector Machine
- `decision-tree`: Decision Tree Classifier

## Script Reference

### train_model.py
```bash
# Basic usage
python scripts/train_model.py --experiment-id my_experiment --feature-strategy minimal

# Advanced options
python scripts/train_model.py \
    --experiment-id advanced_experiment \
    --feature-strategy extended \
    --models random-forest xgboost \
    --cv-folds 10 \
    --random-state 42
```

### evaluate_model.py
```bash
# Evaluate single experiment
python scripts/evaluate_model.py --experiment-id baseline_v1

# Compare multiple experiments
python scripts/evaluate_model.py --compare exp1 exp2 exp3

# Generate detailed analysis
python scripts/evaluate_model.py --experiment-id baseline_v1 --detailed-analysis
```

### predict.py
```bash
# Single prediction
python scripts/predict.py --experiment-id baseline_v1 --input "M,298.1,308.6,1551,42.8,0"

# Batch prediction from CSV
python scripts/predict.py --experiment-id baseline_v1 --input-file data/new_samples.csv --output predictions.csv
```

### generate_report.py
```bash
# Technical report for single experiment
python scripts/generate_report.py --experiment-id baseline_v1 --output reports/baseline_report/

# Comparison report across experiments
python scripts/generate_report.py --compare baseline_v1 extended_v1 --output reports/comparison/
```

## Understanding Results

### Experiment Output Structure
```
models/experiments/
├── baseline_v1/
│   ├── config.json           # Experiment configuration
│   ├── model_random_forest.pkl
│   ├── model_xgboost.pkl
│   ├── scaler.pkl
│   └── feature_list.json
└── extended_v1/
    ├── config.json
    ├── models/
    └── evaluation/
```

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Cross-Validation**: K-fold validation with confidence intervals

### Performance Comparison
```bash
# View experiment comparison table
python scripts/evaluate_model.py --compare baseline_v1 extended_v1 --format table

# Generate performance plots
python scripts/generate_report.py --compare baseline_v1 extended_v1 --plot-performance
```

## Project Structure

```
AI4I/
├── scripts/           # Main executable scripts
│   ├── train_model.py       # Experiment-driven model training
│   ├── evaluate_model.py    # Model evaluation and comparison
│   ├── predict.py           # Prediction pipeline
│   └── generate_report.py   # Technical report generation
├── src/               # Core library code
│   ├── features/            # Feature engineering strategies
│   ├── models/              # Model implementations
│   ├── visualization/       # Plotting and analysis
│   └── utils/               # Configuration and utilities
├── data/              # Dataset storage
│   ├── raw/                 # Original AI4I 2020 dataset
│   ├── processed/           # Preprocessed data per experiment
│   └── features/            # Feature engineering outputs
├── models/            # Trained model artifacts
├── reports/           # Generated technical reports
├── tests/             # Unit tests
└── requirements.txt   # Python dependencies
```

## Configuration

### Experiment Configuration
Create `config/experiment.yaml`:
```yaml
experiment:
  id: "my_experiment_v1"
  description: "Testing extended features with hyperparameter tuning"
  
features:
  strategy: "extended"
  selection:
    method: "mutual_info"
    k_best: 10

models:
  algorithms: ["random_forest", "xgboost"]
  hyperparameter_tuning:
    enabled: true
    n_trials: 50

evaluation:
  cv_folds: 5
  test_size: 0.2
  stratify: true
```

Use with:
```bash
python scripts/train_model.py --config config/experiment.yaml
```

## Testing & Debugging

### Quick Validation
```bash
# Test basic functionality
python scripts/train_model.py --experiment-id test_run --feature-strategy minimal --quick-test

# Should complete without errors and create model artifacts
```

### Common Issues

#### Import Errors
```bash
# Install missing packages
pip install -r requirements.txt

# Check specific package
python -c "import sklearn, pandas, xgboost; print('All packages available')"
```

#### File Path Issues
```bash
# Verify data exists
ls data/raw/ai4i2020.csv

# Check experiment output
ls models/experiments/

# Solution: Complete workflow
python scripts/train_model.py --experiment-id baseline_test
python scripts/evaluate_model.py --experiment-id baseline_test
```

#### Performance Issues
```bash
# Enable debug logging
python scripts/train_model.py --experiment-id debug_test --log-level DEBUG

# Check logs
tail -f logs/train_model.log
```

### Testing Framework
```bash
# Run unit tests
python -m pytest tests/

# Test specific component
python -m pytest tests/test_feature_engineering.py -v
```

## Advanced Usage

### Custom Feature Engineering
Create custom strategy in `src/features/feature_engineering.py`:
```python
def advanced_custom_features(df):
    """Custom feature engineering strategy"""
    # Your domain-specific features
    df['power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    return df
```

### Hyperparameter Optimization
```bash
# Extensive hyperparameter search
python scripts/train_model.py \
    --experiment-id hyperopt_experiment \
    --feature-strategy extended \
    --models xgboost \
    --hyperopt-trials 100 \
    --hyperopt-timeout 3600
```

### Parallel Experiments
```bash
# Run multiple experiments in parallel
python scripts/train_model.py --experiment-id exp1 --feature-strategy minimal &
python scripts/train_model.py --experiment-id exp2 --feature-strategy extended &
python scripts/train_model.py --experiment-id exp3 --feature-strategy advanced &
wait

# Compare all results
python scripts/evaluate_model.py --compare exp1 exp2 exp3
```
