#!/usr/bin/env python3
"""
Complete standalone dashboard with all required API endpoints
"""
import os
import sys
import json
import threading
import time
import webbrowser
import random
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create a FastAPI app for the backend
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model and dataset
model = None
dataset = None
model_name = "Test Model"
model_code = """
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model pipeline
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100, 
        max_depth=None,
        min_samples_split=2,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
"""

# Example improvement suggestions
improvement_suggestions = [
    {
        "title": "Try Feature Selection",
        "description": "Your model has many features. Consider using feature selection to identify and use only the most informative ones.",
        "impact": "high",
        "difficulty": "medium",
        "code_example": """
from sklearn.feature_selection import SelectFromModel

# Use the RandomForest's feature importance for selection
selector = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold='median')
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Train model on selected features
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_selected, y_train)

# Check which features were selected
selected_features = selector.get_support()
selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features[i]]
print(f"Selected {len(selected_feature_names)} features out of {len(feature_names)}")
"""
    },
    {
        "title": "Optimize Hyperparameters",
        "description": "Your model could benefit from hyperparameter tuning using cross-validation.",
        "impact": "medium",
        "difficulty": "medium",
        "code_example": """
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

# Set up grid search
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
"""
    },
    {
        "title": "Address Class Imbalance",
        "description": "Your dataset has an imbalance between classes. Consider using resampling techniques.",
        "impact": "medium",
        "difficulty": "easy",
        "code_example": """
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Create a pipeline with SMOTE
model = ImbPipeline([
    ('scaler', StandardScaler()),
    ('sampler', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
"""
    }
]

# Code generation examples by category
code_examples = {
    "feature_engineering": {
        "sklearn": """
# Feature Engineering with scikit-learn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Create a pipeline with polynomial features
model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

model.fit(X_train, y_train)
""",
        "pytorch": """
# Feature Engineering with PyTorch
import torch
import torch.nn as nn

class FeatureTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x
        
# Use in a model
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature_transformer = FeatureTransformer(input_dim)
        self.classifier = nn.Linear(32, output_dim)
        
    def forward(self, x):
        features = self.feature_transformer(x)
        return self.classifier(features)
""",
        "tensorflow": """
# Feature Engineering with TensorFlow
import tensorflow as tf

def create_feature_layer(input_dim):
    feature_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
    ])
    return feature_layer

# Use in a model
model = tf.keras.Sequential([
    create_feature_layer(input_dim),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
"""
    },
    "regularization": {
        "sklearn": """
# Regularization with scikit-learn
from sklearn.linear_model import LogisticRegression

# L1 regularization (Lasso)
model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)

# L2 regularization (Ridge)
model_l2 = LogisticRegression(penalty='l2', C=0.1)

# ElasticNet (L1 + L2)
model_elastic = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.1)
""",
        "pytorch": """
# Regularization with PyTorch
import torch.nn as nn
import torch.optim as optim

# L2 regularization (weight decay)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# L1 regularization
def l1_loss(model, lambda_l1=0.01):
    l1_reg = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l1_reg = l1_reg + torch.sum(torch.abs(param))
    return lambda_l1 * l1_reg

# Then add to your training loop:
# loss = criterion(outputs, targets) + l1_loss(model)
""",
        "tensorflow": """
# Regularization with TensorFlow
import tensorflow as tf

# L1 regularization
regularizer_l1 = tf.keras.regularizers.l1(0.01)

# L2 regularization
regularizer_l2 = tf.keras.regularizers.l2(0.01)

# L1 + L2 regularization
regularizer_l1_l2 = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)

# Use in a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', 
                         kernel_regularizer=regularizer_l2,
                         input_shape=(input_dim,)),
    tf.keras.layers.Dense(num_classes, activation='softmax',
                         kernel_regularizer=regularizer_l2)
])
"""
    },
    "hardcoded_learning_rate": {
        "sklearn": """
# Learning rate scheduling with scikit-learn
from sklearn.ensemble import GradientBoostingClassifier

# Model with decreasing learning rate
model = GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    max_features='sqrt'
)
""",
        "pytorch": """
# Learning rate scheduling with PyTorch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# Initial optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step learning rate scheduler (decreases by gamma every step_size epochs)
scheduler_step = StepLR(optimizer, step_size=10, gamma=0.1)

# Reduce on plateau (decreases when a metric stops improving)
scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Then in your training loop:
# for epoch in range(epochs):
#     train(...)
#     val_loss = validate(...)
#     scheduler_step.step()  # For StepLR
#     scheduler_plateau.step(val_loss)  # For ReduceLROnPlateau
""",
        "tensorflow": """
# Learning rate scheduling with TensorFlow
import tensorflow as tf

# Exponential decay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9
)

# Optimizer with schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Alternative: callback approach
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    min_lr=0.00001
)

# Use in model.fit:
# model.fit(..., callbacks=[reduce_lr])
"""
    }
}

# API endpoints
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/api/status")
async def status():
    """API status endpoint"""
    return {"status": "connected", "version": "0.1.0"}

@app.get("/api/model")
async def get_model():
    """Get model information"""
    if model is None:
        return {"error": "No model loaded"}
    
    # Get model information
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        estimator = model.named_steps['classifier']
    else:
        estimator = model
    
    # Get feature importances if available
    feature_importances = None
    if hasattr(estimator, 'feature_importances_'):
        feature_importances = estimator.feature_importances_.tolist()
    
    # Basic model info
    model_info = {
        "name": model_name,
        "type": "Classification",
        "framework": "scikit-learn",
        "metrics": {
            "accuracy": float(accuracy_score(dataset['y_test'], model.predict(dataset['X_test']))),
            "precision": float(precision_score(dataset['y_test'], model.predict(dataset['X_test']))),
            "recall": float(recall_score(dataset['y_test'], model.predict(dataset['X_test']))),
            "f1": float(f1_score(dataset['y_test'], model.predict(dataset['X_test'])))
        },
        "dataset": {
            "samples": int(len(dataset['X_train']) + len(dataset['X_test'])),
            "features": int(dataset['X_train'].shape[1]),
            "train_test_split": f"{len(dataset['X_train'])}:{len(dataset['X_test'])}"
        }
    }
    
    # Add feature importance if available
    if feature_importances is not None:
        feature_names = dataset['feature_names'].tolist() if hasattr(dataset['feature_names'], 'tolist') else dataset['feature_names']
        model_info["feature_importances"] = [
            {"name": name, "importance": float(importance)}
            for name, importance in zip(feature_names, feature_importances)
        ]
    
    return model_info

@app.get("/api/model-code")
async def get_model_code():
    """Get the model code"""
    return {"code": model_code}

@app.post("/api/analyze-code")
async def analyze_code(request: Request):
    """Analyze model code"""
    data = await request.json()
    code = data.get("code", "")
    
    # Simulate code analysis
    analysis = {
        "issues": [
            {
                "type": "hardcoded_learning_rate",
                "severity": "medium",
                "description": "The learning rate is hardcoded. Consider using a learning rate scheduler.",
                "line": 15,
                "suggestion": "Implement a learning rate scheduler to adaptively adjust the learning rate during training."
            },
            {
                "type": "missing_regularization",
                "severity": "medium",
                "description": "No regularization is applied to the model. This may lead to overfitting.",
                "line": 10,
                "suggestion": "Add L1 or L2 regularization to prevent overfitting."
            }
        ],
        "performance_score": 75,
        "metrics": {
            "accuracy": 0.86,
            "precision": 0.83,
            "recall": 0.81,
            "f1": 0.82
        }
    }
    
    return analysis

@app.get("/api/model-improvements")
async def get_model_improvements(detail_level: str = "basic"):
    """Get model improvement suggestions"""
    return {"improvements": improvement_suggestions}

@app.get("/api/generate-code-example")
async def generate_code_example(framework: str, category: str):
    """Generate code example for the specified framework and category"""
    if category in code_examples and framework in code_examples[category]:
        return {"code": code_examples[category][framework]}
    else:
        return {"code": "# No example available for this combination"}

@app.get("/api/features")
async def get_features():
    """Get feature information"""
    if dataset is None:
        return {"error": "No dataset loaded"}
    
    feature_names = dataset['feature_names'].tolist() if hasattr(dataset['feature_names'], 'tolist') else dataset['feature_names']
    
    return {
        "features": [
            {"name": name, "type": "numeric"}
            for name in feature_names
        ],
        "num_features": len(feature_names)
    }

@app.get("/api/predictions")
async def get_predictions():
    """Get model predictions"""
    if model is None or dataset is None:
        return {"error": "No model or dataset loaded"}
    
    # Get predictions on test set
    y_pred = model.predict(dataset['X_test'])
    probas = None
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(dataset['X_test'])
    
    # Format response
    results = []
    for i in range(min(20, len(dataset['X_test']))):
        result = {
            "id": i,
            "actual": int(dataset['y_test'][i]),
            "predicted": int(y_pred[i]),
            "correct": bool(dataset['y_test'][i] == y_pred[i])
        }
        
        if probas is not None:
            result["confidence"] = float(probas[i][int(y_pred[i])])
        
        results.append(result)
    
    return {
        "predictions": results,
        "accuracy": float(accuracy_score(dataset['y_test'], y_pred))
    }

def create_model():
    """Create and train a model"""
    global model, dataset, model_name
    
    print("Creating classification model...")
    
    # Load breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline with preprocessing and model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Create dataset dictionary with metadata
    dataset = {
        'X_train': X_train,
        'X_test': X_test, 
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'target_names': data.target_names
    }
    
    model_name = "Breast Cancer Classifier"
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples")

def serve_frontend():
    """Serve frontend from the build directory"""
    repo_root = Path(__file__).parent.absolute()
    frontend_build = repo_root / 'frontend' / 'build'
    
    if not frontend_build.exists():
        print(f"Frontend build not found at: {frontend_build}")
        print("Attempting to build the frontend...")
        
        frontend_dir = repo_root / 'frontend'
        if frontend_dir.exists():
            try:
                os.chdir(frontend_dir)
                subprocess.run(["npm", "install"], check=True)
                subprocess.run(["npm", "run", "build"], check=True)
                
                if not frontend_build.exists():
                    print("Failed to build frontend.")
                    return None
            except Exception as e:
                print(f"Error building frontend: {e}")
                return None
        else:
            print("Frontend directory not found.")
            return None
    
    # Set up static file serving
    app.mount("/", StaticFiles(directory=str(frontend_build), html=True), name="static")
    return True

def main():
    """Main function"""
    # Create model
    create_model()
    
    # Set up frontend serving
    frontend_ok = serve_frontend()
    
    if not frontend_ok:
        print("WARNING: Frontend not available. Only API endpoints will be served.")
        
        @app.get("/")
        async def root():
            return {
                "message": "API is running but frontend is not available. API endpoints:",
                "endpoints": [
                    "/api/health",
                    "/api/status",
                    "/api/model",
                    "/api/features",
                    "/api/predictions",
                    "/api/model-code",
                    "/api/model-improvements",
                    "/api/generate-code-example",
                    "/api/analyze-code"
                ]
            }
    
    # Run the server
    port = 8000
    print(f"\nStarting dashboard on http://localhost:{port}")
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(1)
        webbrowser.open(f"http://localhost:{port}")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()