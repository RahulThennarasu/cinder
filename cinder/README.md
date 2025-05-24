# Cinder ML

Cinder is a powerful tool for ML model debugging and analysis. It provides:

- A dashboard for visualizing model performance
- Automated suggestions for model improvement
- Code generation for fixing common ML issues
- Support for PyTorch, TensorFlow, and scikit-learn models

## Installation

```bash
pip install cinder-ml
```

## Quick Start

```python
import torch
from cinder.model_interface.connector import ModelDebugger

# Load your trained model
model = YourModel()

# Create a test dataloader
test_loader = DataLoader(test_dataset, batch_size=32)

# Connect to Cinder
debugger = ModelDebugger(model, test_loader, name="My Model")

# Analyze the model
results = debugger.analyze()

# Print analysis
print(f"Accuracy: {results['accuracy']*100:.2f}%")
print(f"Errors: {results['error_analysis']['error_count']}")

# Launch the dashboard
debugger.launch_dashboard()
```

The dashboard will be available at http://localhost:8000

## Features

- **Model Analysis**: Get comprehensive metrics on your model's performance
- **Error Analysis**: Identify common error patterns
- **Improvement Suggestions**: Get actionable suggestions to improve your model
- **Interactive Dashboard**: Visualize your model's performance with an interactive dashboard
- **Code Generation**: Generate code to implement the suggested improvements

## Examples

Check out the examples directory for more usage examples:

- Basic classification with PyTorch
- Image classification with MNIST
- Handling class imbalance
- And more!

## License

MIT