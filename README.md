# Cinder

Cinder is a comprehensive machine learning model debugging and analysis tool designed to provide visual insights, performance metrics, and improvement suggestions for ML models. It supports multiple frameworks including PyTorch, TensorFlow, and scikit-learn.

## Installation

```bash
pip install cinder-ml
```

For additional framework support:

```bash
pip install "cinder-ml[pytorch]"    # PyTorch support
pip install "cinder-ml[tensorflow]" # TensorFlow support
pip install "cinder-ml[all]"        # All frameworks
```

## API Key Authentication

Starting from version 1.1.0, Cinder requires an API key for authentication.

### Getting an API Key

You can generate an API key using the CLI:

```bash
cinder generate-key --user-id your_username

## Features

- Interactive visual dashboard for model analysis
- Comprehensive performance metrics and error analysis
- Confusion matrix visualization
- Feature importance analysis
- Error type categorization
- Model improvement suggestions with code examples
- Training history visualization
- Cross-validation analysis
- Support for PyTorch, TensorFlow, and scikit-learn models

## Quick Start

```python
from cinder import ModelDebugger
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate and prepare data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Create a dataset wrapper for Cinder
class DatasetWrapper:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.current = 0
    
    def __iter__(self):
        self.current = 0
        return self
    
    def __next__(self):
        if self.current >= len(self.X):
            raise StopIteration
        X_batch = self.X[self.current:self.current+32]
        y_batch = self.y[self.current:self.current+32]
        self.current += 32
        return X_batch, y_batch

# Create a dataset for analysis
dataset = DatasetWrapper(X_test, y_test)

# Initialize Cinder's ModelDebugger
debugger = ModelDebugger(model, dataset, name="Classification Model")

# Run analysis
results = debugger.analyze()
print(f"Model accuracy: {results['accuracy']:.4f}")

# Launch the dashboard
debugger.launch_dashboard()
```

Visit http://localhost:8000 in your browser to access the dashboard.

## Framework-Specific Usage

### scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
from cinder import ModelDebugger

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Initialize Cinder
debugger = ModelDebugger(model, dataset, name="Random Forest")
debugger.launch_dashboard()
```

### PyTorch

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cinder import ModelDebugger

# Define and train model
model = YourPyTorchModel()
# ... training code ...

# Use with a DataLoader
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize Cinder
debugger = ModelDebugger(model, test_loader, name="PyTorch Model")
debugger.launch_dashboard()
```

### TensorFlow

```python
import tensorflow as tf
from cinder import ModelDebugger

# Define and train model
model = tf.keras.Sequential([
    # ... model layers ...
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=10)

# Create dataset wrapper
class TFDataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.current = 0
    
    def __iter__(self):
        self.current = 0
        return self
    
    def __next__(self):
        if self.current >= len(self.x):
            raise StopIteration
        x_batch = self.x[self.current:self.current+32]
        y_batch = self.y[self.current:self.current+32]
        self.current += 32
        return x_batch, y_batch

# Initialize Cinder
dataset = TFDataset(X_test, y_test)
debugger = ModelDebugger(model, dataset, name="TensorFlow Model")
debugger.launch_dashboard()
```

## Advanced Usage

### Analyzing Model Performance

```python
# Run comprehensive analysis
results = debugger.analyze()

# Key metrics
accuracy = results['accuracy']
precision = results['precision']
recall = results['recall']
f1 = results['f1']

# Error analysis
error_analysis = results['error_analysis']
error_count = error_analysis['error_count']
error_rate = error_analysis['error_rate']

# Confusion matrix
confusion_matrix = results['confusion_matrix']
```

### Getting Improvement Suggestions

```python
# Get improvement suggestions
suggestions = debugger.get_improvement_suggestions(detail_level="comprehensive")

# Print top suggestions
for suggestion in suggestions["suggestions"]:
    print(f"- {suggestion['title']}: {suggestion['suggestion']}")
```

### Analyzing Feature Importance

```python
# Get feature importance
importance = debugger.analyze_feature_importance()

# Print top features
for i, (feature, value) in enumerate(zip(importance['feature_names'], importance['importance_values'])):
    if i < 5:  # Top 5 features
        print(f"{feature}: {value:.4f}")
```

### Tracking Training History

```python
# Add training history data
history = [
    {"iteration": 1, "accuracy": 0.75, "loss": 0.35},
    {"iteration": 2, "accuracy": 0.82, "loss": 0.28},
    # ... more epochs ...
]
debugger.training_history = history
```

### Performing Cross-Validation

```python
# Perform cross-validation
cv_results = debugger.perform_cross_validation(k_folds=5)

# Print cross-validation results
print(f"Mean accuracy: {cv_results['mean_accuracy']:.4f}")
print(f"Standard deviation: {cv_results['std_accuracy']:.4f}")
```

### Customizing Dashboard

```python
# Launch dashboard on a specific port
debugger.launch_dashboard(port=9000)
```

## Command Line Interface

Cinder includes a command-line interface for quick access to examples:

```bash
# Show help
cinder --help

# Run examples
cinder run quickstart
cinder run pytorch
cinder run sklearn
cinder run tensorflow

# Start the dashboard server directly
cinder serve --port 8000
```

## Technical Documentation

### ModelDebugger Class

The main interface to Cinder is the ModelDebugger class:

```python
ModelDebugger(model, dataset, name=None)
```

Parameters:
- `model`: A trained ML model (PyTorch, TensorFlow, or scikit-learn)
- `dataset`: A dataset that yields (input, target) pairs
- `name`: Optional name for the model

Methods:
- `analyze()`: Run comprehensive analysis on the model
- `launch_dashboard(port=8000)`: Start the dashboard server
- `analyze_confidence()`: Analyze prediction confidence
- `analyze_feature_importance()`: Analyze feature importance
- `get_improvement_suggestions(detail_level="comprehensive")`: Get improvement suggestions
- `perform_cross_validation(k_folds=5)`: Perform cross-validation
- `analyze_prediction_drift(threshold=0.1)`: Analyze prediction drift
- `get_sample_predictions(limit=10, offset=0, errors_only=False)`: Get sample predictions

### Dataset Format

Cinder expects datasets to implement the iterator protocol, yielding (inputs, targets) pairs:

```python
class YourDataset:
    def __iter__(self):
        # Initialize iteration
        return self
    
    def __next__(self):
        # Return next batch of (inputs, targets)
        # Raise StopIteration when done
        if no_more_data:
            raise StopIteration
        return inputs_batch, targets_batch
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

Cinder uses several open source libraries including FastAPI, scikit-learn, and React.
