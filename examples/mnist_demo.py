#!/usr/bin/env python3
"""
CompileML Demo - MNIST Digit Classification

This script trains a CNN on the Fashion MNIST dataset and connects it to CompileML
for analysis and debugging.

Requirements:
- PyTorch
- torchvision
- CompileML (your backend code)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import threading
import time

# Add the parent directory to the path to import CompileML
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import CompileML components
from cinder.cinder.model_interface.connector import ModelDebugger

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define a CNN model for MNIST/Fashion MNIST
class FashionMNISTNet(nn.Module):
    def __init__(self):
        super(FashionMNISTNet, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        # Dropout for regularization
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # Output layer (no softmax - it's included in loss function)
        return x

def train_model(model, train_loader, optimizer, device, epochs=3):
    """Train the model on the training data."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = F.cross_entropy(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Acc: {100.*correct/total:.2f}%')
                
        # Print epoch statistics
        print(f'Epoch: {epoch+1}/{epochs} | '
              f'Loss: {total_loss/len(train_loader):.4f} | '
              f'Acc: {100.*correct/total:.2f}%')

def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test data."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    print(f'Test Accuracy: {100. * accuracy:.2f}%')
    return accuracy

def launch_dashboard_thread(debugger):
    """Launch dashboard in a separate thread."""
    debugger.launch_dashboard()

def main():
    """Main function to train and analyze the model."""
    print("Fashion MNIST Classification with CompileML")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load Fashion MNIST dataset instead of MNIST (more reliable download)
    print("Loading Fashion MNIST dataset...")
    try:
        # Try with download set to True
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    except (RuntimeError, Exception) as e:
        print(f"Error downloading dataset: {e}")
        print("Checking if dataset already exists...")
        # Try without download flag
        train_dataset = datasets.FashionMNIST('./data', train=True, download=False, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, download=False, transform=transform)
    
    # Define class names for Fashion MNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print(f"Dataset loaded with {len(train_dataset)} training and {len(test_dataset)} test samples")
    print(f"Classes: {class_names}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Create a smaller test dataset for quick analysis
    small_test, _ = random_split(test_dataset, [1000, len(test_dataset)-1000])
    small_test_loader = DataLoader(small_test, batch_size=64)
    
    # Initialize the model
    print("Creating model...")
    model = FashionMNISTNet().to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Training model...")
    train_model(model, train_loader, optimizer, device, epochs=3)
    
    # Evaluate the model
    print("Evaluating model...")
    evaluate_model(model, test_loader, device)
    
    # Connect to CompileML for analysis
    print("Connecting to CompileML...")
    debugger = ModelDebugger(model, small_test_loader, name="Fashion MNIST CNN")
    
    # Run analysis
    print("Running analysis...")
    results = debugger.analyze()
    
    # Print analysis results
    print("\nCompileML Analysis Results:")
    print(f"- Model: {results['model_name']} ({results['framework']})")
    print(f"- Dataset size: {results['dataset_size']} samples")
    print(f"- Accuracy: {results['accuracy']*100:.2f}%")
    if 'precision' in results:
        print(f"- Precision: {results['precision']*100:.2f}%")
    if 'recall' in results:
        print(f"- Recall: {results['recall']*100:.2f}%")
    if 'f1' in results:
        print(f"- F1 Score: {results['f1']*100:.2f}%")
    print(f"- Errors: {results['error_analysis']['error_count']} ({results['error_analysis']['error_rate']*100:.2f}%)")
    
    # Launch the CompileML dashboard in a separate thread
    print("\nLaunching CompileML dashboard...")
    print("Access the dashboard at http://localhost:8000")
    print("Press Ctrl+C to exit")
    
    # Start the dashboard in a separate thread
    dashboard_thread = threading.Thread(
        target=launch_dashboard_thread,
        args=(debugger,),
        daemon=False  # Use non-daemon thread so it keeps running
    )
    dashboard_thread.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)  # Sleep to avoid high CPU usage
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()