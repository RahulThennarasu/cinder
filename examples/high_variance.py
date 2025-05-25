import sys
import os
# Add the parent directory to the path so we can import the backend module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from cinder import ModelDebugger

# Define an overly complex model to cause overfitting
class ComplexNeuralNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=256, num_classes=2):  # Very large hidden layer
        super(ComplexNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)  # Fixed: takes hidden_size input
        self.layer3 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer4 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)  # Fixed: using out instead of x
        out = self.relu(out)
        out = self.layer3(out)
        out = self.relu(out)
        out = self.layer4(out)
        return F.log_softmax(out, dim=1)

def generate_synthetic_data(num_samples=100, input_size=10, num_classes=2, train_ratio=0.8):
    """Generate a very small synthetic dataset to encourage overfitting."""
    # Create features with clear class separation for better visualization
    X = torch.randn(num_samples, input_size)
    
    # Create a deterministic decision boundary for classification
    weights = torch.randn(input_size)
    bias = torch.randn(1)
    
    # Compute raw scores
    scores = torch.matmul(X, weights) + bias
    
    # Convert to binary labels (0 or 1)
    y = (scores > 0).long()
    
    # Split into train and test
    train_size = int(num_samples * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    return train_dataset, test_dataset

def train_model(model, train_loader, num_epochs=100):  # Many epochs to encourage overfitting
    """Train the model with the provided data."""
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Add a dictionary to store training history
    history = {
        'accuracy': [],
        'loss': [],
    }
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate epoch accuracy
        epoch_accuracy = correct / total
        epoch_loss = total_loss / len(train_loader)
        
        # Store in history
        history['accuracy'].append(epoch_accuracy)
        history['loss'].append(epoch_loss)
        
        # Print epoch stats every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {100*epoch_accuracy:.2f}%')
    
    return history

def main():
    print("Generating small synthetic dataset to encourage overfitting...")
    train_dataset, test_dataset = generate_synthetic_data(
        num_samples=100,  # Small dataset
        input_size=10, 
        num_classes=2
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Small batch size
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print("Creating complex model to cause overfitting...")
    model = ComplexNeuralNetwork(input_size=10, hidden_size=256, num_classes=2)  # Overly complex
    
    print("Training model for many epochs...")
    history = train_model(model, train_loader, num_epochs=100)  # Many epochs
    
    print("Initializing ModelDebugger...")
    # Connect the model to CompileML
    debugger = ModelDebugger(model, test_loader, name="Overfitting Neural Network")
    
    # Add training history to debugger
    # Make training accuracy artificially high to make overfitting more obvious
    train_accuracy = [min(acc * 1.2, 0.99) for acc in history['accuracy']]
    debugger.training_history = [
        {"iteration": i+1, "accuracy": acc, "loss": loss} 
        for i, (acc, loss) in enumerate(zip(train_accuracy, history['loss']))
    ]
    
    # Override confidence values to indicate overfitting
    def override_confidence_analysis(self):
        return {
            "avg_confidence": 0.88,
            "avg_correct_confidence": 0.95,  # Very high confidence on correct predictions
            "avg_incorrect_confidence": 0.75,  # Also high confidence on incorrect predictions (overconfident)
            "calibration_error": 0.20,  # Large error
            "confidence_distribution": {
                "bin_edges": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "overall": [0, 0, 0, 0, 0, 5, 10, 15, 30, 40],  # Most predictions have high confidence
                "correct": [0, 0, 0, 0, 0, 0, 5, 10, 25, 35],   # Correct predictions have very high confidence
                "incorrect": [0, 0, 0, 0, 0, 5, 5, 5, 5, 5],    # Some incorrect have high confidence too (bad)
            },
            "overconfident_examples": {
                "threshold": 0.9,
                "count": 4,
                "indices": [1, 5, 8, 12]
            },
            "underconfident_examples": {
                "threshold": 0.6,
                "count": 2,
                "indices": [3, 9]
            }
        }
    
    # Override the analyze_confidence method to return our fixed values
    debugger.analyze_confidence = lambda: override_confidence_analysis(debugger)
    
    print("Running analysis...")
    # Run analysis
    results = debugger.analyze()
    print(f"Analysis results:")
    print(f"  - Model: {results['model_name']} ({results['framework']})")
    print(f"  - Dataset size: {results['dataset_size']} samples")
    print(f"  - Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  - Errors: {results['error_analysis']['error_count']} ({results['error_analysis']['error_rate']*100:.2f}%)")
    
    print("Launching dashboard...")
    # Launch the debugging dashboard
    debugger.launch_dashboard()
    
    # Keep the server running
    try:
        print("Dashboard running at http://localhost:8000")
        print("Press Ctrl+C to exit...")
        # This will keep the main thread running until interrupted
        while True:
            pass
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()