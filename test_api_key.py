from cinder import ModelDebugger
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Replace this with an API key you generated from your dashboard
API_KEY = "cinder_1748479894241_i9fsbitv79"

print(f"Testing API key: {API_KEY}")

# Create a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

# Create dummy data
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32)

# Initialize the model
model = SimpleModel()

# Initialize ModelDebugger with your API key
try:
    print("Initializing ModelDebugger with API key...")
    debugger = ModelDebugger(model, dataloader, name="Test Model", api_key=API_KEY)
    print("✓ API key validation successful!")
    
    # Run analysis
    print("Running analysis...")
    results = debugger.analyze()
    print(f"✓ Analysis complete! Accuracy = {results['accuracy']:.4f}")
    
    # Test a few more requests for rate limiting
    print("\nTesting rate limits (for free tier: 100 requests/day)...")
    for i in range(105):  # Try to exceed the free tier limit
        try:
            results = debugger.analyze()
            print(f"  Request {i+1}: Success - Accuracy = {results['accuracy']:.4f}")
        except Exception as e:
            print(f"  Request {i+1}: Failed - {e}")
            break
        
        # Add a small delay to avoid overwhelming Firebase
        import time
        time.sleep(0.1)
        
except ValueError as e:
    print(f"✗ API key validation failed: {e}")
except Exception as e:
    print(f"✗ Error during testing: {e}")