
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
Then use X_scaled to create train and test datasets.

```python
# After scaling, use X_scaled for creating datasets
train_dataset = TensorDataset(torch.tensor(X_scaled[:train_size]).float(), y_train)
test_dataset = TensorDataset(torch.tensor(X_scaled[train_size:]).float(), y_test)
```

X_scaled = scaler.fit_transform(X)

```python
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        return F.log_softmax(out, dim=1)
```

```python
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        return F.log_softmax(out, dim=1)
```



```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Should be removed, or `X` replaced with `X_scaled` if data scaling is intended.


```python
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_classes=2, dropout_rate=0.2): # Added dropout_rate

```python
def train_model(model, train_loader, test_loader, num_epochs=10):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')

        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test
        print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

        scheduler.step(avg_test_loss)
```

        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)

```python
        # Step the scheduler at the end of each epoch
        scheduler.step(total_loss/len(train_loader))
```

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) # Added dropout layer
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)

```python
def main():
    # ... other code ...

    print("Training model...")
    train_model(model, train_loader, test_loader, num_epochs=5)

    print("Evaluating model...")
    model.eval()
    # ... evaluate the model on the test set using test_loader ...
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f} %')
    model.train()
    # ... more code ...
```


```python
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_classes=2, dropout_rate=0.2):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        return F.log_softmax(out, dim=1)

```python
def train_model(model, train_loader, test_loader, num_epochs=10):
    criterion = nn.NLLLoss()

# Use learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')

        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test
        print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

        scheduler.step(avg_test_loss)
```
Modified the `train_model` function to account for the change:
```python
train_model(model, train_loader, test_loader, num_epochs=5)
```

```

        out = self.dropout(out) # Apply dropout
        out = self.layer2(out)
        return F.log_softmax(out, dim=1)
```



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

import sys
import os
# Add the parent directory to the path so we can import the backend module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


```python
# Remove the redudant instances of the learning rate scheduler.
```

```python
class NeuralNetwork(nn.Module):

```python
model.eval()  # Set model to evaluation mode

```python
def train_model(model, train_loader, num_epochs=10):
    criterion = nn.NLLLoss()

# Use learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)


# Use learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
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
        
        # Print epoch stats
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')
        scheduler.step(total_loss/len(train_loader))
```

with torch.no_grad():
    predictions = model(input_data)

    # Compute raw scores
    scores = torch.matmul(X, weights) + bias
    
    # Convert to binary labels (0 or 1)
    y = (scores > 0).long()
    
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


with torch.no_grad():
    predictions = model(input_data)
```

    def __init__(self, input_size=10, hidden_size=20, num_classes=2, dropout_rate=0.2): # Added dropout_rate
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) # Added dropout layer
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out) # Apply dropout
        out = self.layer2(out)
        return F.log_softmax(out, dim=1)
```

import numpy as np

from backend.model_interface.connector import ModelDebugger

# Define a more complex model than the original simple example
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_classes=2):
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
    """Generate synthetic data for demonstration."""

```python
def train_model(model, train_loader, num_epochs=10):
    """Train the model with the provided data."""
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Use learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)


# Use learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

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
        
        # Step the scheduler at the end of each epoch
        scheduler.step(total_loss/len(train_loader))

        # Print epoch stats
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')
```

    # Create features with clear class separation for better visualization
    X = torch.randn(num_samples, input_size)
    
    # Create a deterministic decision boundary for classification
    weights = torch.randn(input_size)
    bias = torch.randn(1)
    

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    predictions = model(input_data)

    # Compute raw scores
    scores = torch.matmul(X, weights) + bias
    
    # Convert to binary labels (0 or 1)

# Use learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    y = (scores > 0).long()
    
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

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    predictions = model(input_data)


# Use learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    
    return train_dataset, test_dataset


with torch.no_grad():
    predictions = model(input_data)

# Use learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)


# Use learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)


def train_model(model, train_loader, num_epochs=10):
    """Train the model with the provided data."""
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
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
        
        # Print epoch stats
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')

def main():
    print("Generating synthetic dataset...")
    train_dataset, test_dataset = generate_synthetic_data(
        num_samples=500, 
        input_size=10, 
        num_classes=2
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("Creating model...")
    model = NeuralNetwork(input_size=10, hidden_size=20, num_classes=2)
    
    print("Training model...")
    train_model(model, train_loader, num_epochs=5)
    
    print("Initializing ModelDebugger...")
    # Connect the model to CompileML
    debugger = ModelDebugger(model, test_loader, name="Neural Network Classifier")
    
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