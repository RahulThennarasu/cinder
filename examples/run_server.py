import sys
import os
# Add the parent directory to the path so we can import the backend module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from backend.model_interface.connector import ModelDebugger

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)

# Create a simple dataset
X = torch.randn(100, 10)  # 100 samples, 10 features
y = torch.randint(0, 2, (100,))  # Binary labels
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10)

# Create model
model = SimpleModel()

# Initialize debugger
debugger = ModelDebugger(model, dataloader, "Test Model")

# Run analysis
results = debugger.analyze()
print(f"Analysis results: {results}")

# Start server directly
from backend.app.server import start_server

if __name__ == "__main__":
    print("Starting server...")
    start_server(debugger, port=8000)