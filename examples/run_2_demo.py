import sys
import os
# Add the parent directory to the path so we can import the backend module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from cinder.cinder.model_interface.connector import ModelDebugger

# Define a more complex model than the original simple example
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=5, num_classes=2):  # Reduced hidden_size to 5 to create underfitting
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return F.log_softmax(out, dim=1)

def generate_synthetic_data(num_samples=500, input_size=10, num_classes=2, train_ratio=0.8):
    """Generate synthetic data for demonstration with a severe class imbalance."""
    # Create features with clear class separation for better visualization
    X = torch.randn(num_samples, input_size)
    
    # Create a deterministic decision boundary for classification
    weights = torch.randn(input_size)
    bias = torch.randn(1)
    
    # Compute raw scores
    scores = torch.matmul(X, weights) + bias
    
    # Convert to binary labels (0 or 1)
    y = (scores > 0).long()
    
    # Create class imbalance (90% class 0, 10% class 1)
    imbalance_mask = torch.rand(num_samples) < 0.8
    y[imbalance_mask] = 0
    
    # Add some noise (flip 10% of the labels)
    noise_indices = torch.randperm(num_samples)[:int(num_samples * 0.1)]
    y[noise_indices] = 1 - y[noise_indices]
    
    # Split into train and test
    train_size = int(num_samples * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    return train_dataset, test_dataset

def train_model(model, train_loader, num_epochs=2):  # Reduced epochs for underfitting
    """Train the model with the provided data."""
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Low learning rate
    
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
        
        # Print epoch stats
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {100*epoch_accuracy:.2f}%')
    
    return history

def main():
    print("Generating synthetic dataset with class imbalance...")
    train_dataset, test_dataset = generate_synthetic_data(
        num_samples=500, 
        input_size=10, 
        num_classes=2
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("Creating simplified model...")
    model = NeuralNetwork(input_size=10, hidden_size=5, num_classes=2)  # Smaller model
    
    print("Training model with reduced epochs...")
    history = train_model(model, train_loader, num_epochs=2)  # Underfitting
    
    print("Initializing ModelDebugger...")
    # Connect the model to CompileML
    debugger = ModelDebugger(model, test_loader, name="Problematic Neural Network")
    
    # Add training history to debugger
    debugger.training_history = [
        {"iteration": i+1, "accuracy": acc, "loss": loss} 
        for i, (acc, loss) in enumerate(zip(history['accuracy'], history['loss']))
    ]
    
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