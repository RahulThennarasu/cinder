import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.model_interface.connector import ModelDebugger

# Define a simple model (same as before)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main():
    print("Creating synthetic dataset...")
    # Create a synthetic dataset instead of downloading MNIST
    num_samples = 100
    
    # Create random 28x28 images (100 samples)
    images = torch.randn(num_samples, 1, 28, 28)
    
    # Create random labels (0-9)
    labels = torch.randint(0, 10, (num_samples,))
    
    # Create a PyTorch dataset and dataloader
    synthetic_dataset = TensorDataset(images, labels)
    dataloader = DataLoader(synthetic_dataset, batch_size=10)
    
    print("Creating model...")
    # Create a model
    model = SimpleNN()
    
    print("Initializing ModelDebugger...")
    # Connect the model to CompileML
    debugger = ModelDebugger(model, dataloader, name="Synthetic Example")
    
    print("Running analysis...")
    # Run analysis
    results = debugger.analyze()
    print(f"Analysis results: {results}")
    
    print("Launching dashboard...")
    # Launch the debugging dashboard
    debugger.launch_dashboard()
    
    # Keep the server running
    try:
        print("Dashboard running at http://localhost:8000")
        print("Press Enter to exit...")
        input()
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()