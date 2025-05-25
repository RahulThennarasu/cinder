# example.py
from cinder import ModelDebugger
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create a simple dataset wrapper
class SimpleDataset:
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

# Create a dataset
dataset = SimpleDataset(X_test, y_test)

# Initialize ModelDebugger
debugger = ModelDebugger(model, dataset, name="Example Model")

# Run analysis
results = debugger.analyze()
print(f"Model accuracy: {results['accuracy']:.4f}")

# Launch dashboard
debugger.launch_dashboard()
print("Dashboard is running at http://localhost:8000")
print("Press Ctrl+C to exit")

# Keep the server running
try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")