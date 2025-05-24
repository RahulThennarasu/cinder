#!/usr/bin/env python3
"""
Standalone server example that doesn't rely on the existing package structure.
This should work regardless of your current setup.
"""
import asyncio
import logging
import uvicorn
import numpy as np
import time
import os
from typing import Dict, Any, List, Optional

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("standalone-server")

# Create a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_classes=2):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return F.log_softmax(out, dim=1)

# Generate synthetic data
def generate_data(num_samples=500, input_size=10):
    """Generate synthetic data for demonstration."""
    X = torch.randn(num_samples, input_size)
    weights = torch.randn(input_size)
    bias = torch.randn(1)
    scores = torch.matmul(X, weights) + bias
    y = (scores > 0).long()
    
    # Add some noise
    noise_indices = torch.randperm(num_samples)[:int(num_samples * 0.1)]
    y[noise_indices] = 1 - y[noise_indices]
    
    return X, y

# Train the model
def train_model(model, X, y, num_epochs=5):
    """Train the model with the provided data."""
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Store training history
    history = []
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
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
        
        # Calculate accuracy for this epoch
        accuracy = correct / total
        
        # Add to history
        history.append({
            'iteration': epoch + 1,
            'accuracy': accuracy,
            'loss': total_loss / len(dataloader),
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Print epoch stats
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {100*accuracy:.2f}%')
    
    return model, history

# Track server start time
server_start_time = datetime.now()

# Create the FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model and data
model = None
X_test = None
y_test = None
predictions = None
ground_truth = None
framework = "pytorch"
model_name = "Simple PyTorch Model"
dataset_size = 0
training_history = []

# API endpoints
@app.get("/api/status")
async def get_status():
    """Get server status."""
    global server_start_time
    
    # Calculate uptime
    uptime = datetime.now() - server_start_time
    uptime_str = str(timedelta(seconds=int(uptime.total_seconds())))
    
    return {
        "status": "online",
        "uptime": uptime_str,
        "connected_model": model_name,
        "memory_usage": 300.0,  # Mock memory usage
        "version": "1.0.0",
        "started_at": server_start_time.isoformat()
    }

@app.get("/api/model")
async def get_model_info():
    """Get model information."""
    global model, framework, model_name, dataset_size, predictions, ground_truth
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    # Calculate accuracy
    accuracy = 0.0
    if predictions is not None and ground_truth is not None:
        accuracy = float(np.mean(predictions == ground_truth))
    
    return {
        "name": model_name,
        "framework": framework,
        "dataset_size": dataset_size,
        "accuracy": accuracy,
        "precision": 0.8,  # Mock value
        "recall": 0.8,  # Mock value
        "f1": 0.8  # Mock value
    }

@app.get("/api/model-code")
async def get_model_code():
    """Get model code."""
    global model
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    # Return a simple model code
    code = """import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_classes=2):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return F.log_softmax(out, dim=1)
"""
    
    return {
        "code": code,
        "file_path": "model.py",
        "framework": framework
    }

@app.get("/api/confusion-matrix")
async def get_confusion_matrix():
    """Get confusion matrix."""
    global predictions, ground_truth
    
    # Check if predictions and ground truth are available
    if predictions is None or ground_truth is None:
        raise HTTPException(status_code=404, detail="No predictions available")
    
    # Create a simple confusion matrix
    classes = np.unique(np.concatenate((predictions, ground_truth)))
    n_classes = len(classes)
    
    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Fill confusion matrix
    for i in range(len(predictions)):
        cm[ground_truth[i], predictions[i]] += 1
    
    return {
        "matrix": cm.tolist(),
        "labels": [str(c) for c in classes],
        "num_classes": int(n_classes)
    }

@app.get("/api/errors")
async def get_errors():
    """Get error analysis."""
    global predictions, ground_truth
    
    # Check if predictions and ground truth are available
    if predictions is None or ground_truth is None:
        raise HTTPException(status_code=404, detail="No predictions available")
    
    # Calculate errors
    error_mask = predictions != ground_truth
    error_indices = np.where(error_mask)[0]
    error_rate = len(error_indices) / len(ground_truth)
    
    return {
        "error_count": len(error_indices),
        "correct_count": len(ground_truth) - len(error_indices),
        "error_rate": float(error_rate),
        "error_indices": error_indices.tolist(),
        "error_types": []
    }

@app.get("/api/training-history")
async def get_training_history():
    """Get training history."""
    global training_history
    
    return training_history

@app.get("/api/model-improvements")
async def get_model_improvements(detail_level: str = Query("comprehensive")):
    """Get model improvement suggestions."""
    global model
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    # Return mock improvement suggestions
    return {
        "model_accuracy": 0.8,
        "error_rate": 0.2,
        "improvement_potential": "medium",
        "suggestions": [
            {
                "category": "data_preparation",
                "title": "Address Class Imbalance",
                "issue": "Class imbalance detected with 70.0% of samples in class 1",
                "suggestion": "Use class weighting or resampling techniques to balance your dataset",
                "severity": "high",
                "expected_impact": "medium-high"
            },
            {
                "category": "regularization",
                "title": "Add Regularization",
                "issue": "Model may be overfitting",
                "suggestion": "Add dropout layers to reduce overfitting",
                "severity": "medium",
                "expected_impact": "medium"
            }
        ]
    }

@app.get("/api/prediction-distribution")
async def get_prediction_distribution():
    """Get prediction distribution."""
    global predictions
    
    # Check if predictions are available
    if predictions is None:
        raise HTTPException(status_code=404, detail="No predictions available")
    
    # Calculate class distribution
    unique_classes = np.unique(predictions)
    distribution = []
    
    for cls in unique_classes:
        count = np.sum(predictions == cls)
        distribution.append({
            "class_name": f"Class {cls}",
            "count": int(count)
        })
    
    return distribution

@app.get("/api/feature-importance")
async def get_feature_importance():
    """Get feature importance."""
    global model
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    # Create mock feature importance
    return {
        "feature_names": [f"Feature {i}" for i in range(10)],
        "importance_values": [0.2, 0.15, 0.12, 0.1, 0.1, 0.08, 0.08, 0.07, 0.05, 0.05],
        "importance_method": "gradient_based"
    }

@app.get("/api/confidence-analysis")
async def get_confidence_analysis():
    """Get confidence analysis."""
    global predictions, ground_truth
    
    # Check if predictions and ground truth are available
    if predictions is None or ground_truth is None:
        raise HTTPException(status_code=404, detail="No predictions available")
    
    # Create mock confidence analysis
    return {
        "avg_confidence": 0.8,
        "avg_correct_confidence": 0.9,
        "avg_incorrect_confidence": 0.6,
        "calibration_error": 0.1,
        "confidence_distribution": {
            "bin_edges": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "overall": [0, 0, 0, 0, 10, 20, 30, 40, 150, 250],
            "correct": [0, 0, 0, 0, 5, 10, 20, 30, 140, 245],
            "incorrect": [0, 0, 0, 0, 5, 10, 10, 10, 10, 5]
        },
        "overconfident_examples": {
            "threshold": 0.9,
            "count": 5,
            "indices": [10, 20, 30, 40, 50]
        },
        "underconfident_examples": {
            "threshold": 0.6,
            "count": 5,
            "indices": [15, 25, 35, 45, 55]
        }
    }


def make_predictions(model, X, y):
    """Make predictions using the model."""
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, preds = torch.max(outputs, 1)
    
    return preds.numpy(), y.numpy()

def run_server(host="0.0.0.0", port=8000):
    """Run the server."""
    global model, X_test, y_test, predictions, ground_truth, training_history, dataset_size
    
    # Create model
    print("Creating model...")
    model = SimpleModel()
    
    # Generate data
    print("Generating data...")
    X, y = generate_data(num_samples=500)
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    print("Training model...")
    model, history = train_model(model, X_train, y_train)
    
    # Store training history
    training_history = history
    
    # Make predictions
    print("Making predictions...")
    predictions, ground_truth = make_predictions(model, X_test, y_test)
    
    # Set dataset size
    dataset_size = len(X_test)
    
    # Mount static files if available
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    if os.path.exists(static_dir):
        print(f"Mounting static files from {static_dir}")
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    else:
        print(f"Static directory not found at {static_dir}")
        print("The dashboard UI may not be available.")
        
        # Create a simple HTML response for the root path
        @app.get("/")
        async def get_root():
            return {
                "message": "ML Dashboard API is running",
                "endpoints": [
                    "/api/status",
                    "/api/model",
                    "/api/model-code",
                    "/api/confusion-matrix",
                    "/api/errors",
                    "/api/training-history",
                    "/api/model-improvements",
                    "/api/prediction-distribution",
                    "/api/feature-importance",
                    "/api/confidence-analysis"
                ]
            }
    
    # Run the server
    print(f"Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()