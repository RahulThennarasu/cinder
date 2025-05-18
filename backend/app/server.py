import asyncio
import logging
import uvicorn
import random
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="CompileML API")

# Enable CORS - this is critical for connecting your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store reference to the ModelDebugger instance
debugger = None

# Store training history (this would be part of the ModelDebugger in a real app)
training_history = []

# API Models
class ModelInfoResponse(BaseModel):
    name: str
    framework: str
    dataset_size: int
    accuracy: float

class ErrorType(BaseModel):
    name: str
    value: int

class TrainingHistoryItem(BaseModel):
    iteration: int
    accuracy: float
    loss: Optional[float] = None
    timestamp: Optional[str] = None

class PredictionDistributionItem(BaseModel):
    class_name: str
    count: int

class ConfusionMatrixResponse(BaseModel):
    matrix: List[List[int]]
    labels: List[str]

class ErrorAnalysisResponse(BaseModel):
    error_count: int
    correct_count: int
    error_indices: List[int]

class ServerStatusResponse(BaseModel):
    status: str
    uptime: str
    connected_model: Optional[str] = None
    memory_usage: Optional[float] = None

# Root endpoint
@app.get("/")
async def root():
    return {"message": "CompileML API is running"}

# Status endpoint
@app.get("/api/status", response_model=ServerStatusResponse)
async def get_status():
    global debugger
    return {
        "status": "online",
        "uptime": "1h 23m", # This would be calculated in a real app
        "connected_model": debugger.name if debugger else None,
        "memory_usage": random.uniform(200, 500) # Mock memory usage in MB
    }

# Model info endpoint
@app.get("/api/model", response_model=ModelInfoResponse)
async def get_model_info():
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    return {
        "name": debugger.name,
        "framework": debugger.framework,
        "dataset_size": len(debugger.ground_truth) if debugger.ground_truth is not None else 0,
        "accuracy": debugger.analyze()["accuracy"],
    }

# Error analysis endpoint
@app.get("/api/errors", response_model=ErrorAnalysisResponse)
async def get_errors():
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    analysis = debugger.analyze()
    error_indices = analysis["error_analysis"]["error_indices"]
    
    return {
        "error_count": analysis["error_analysis"]["error_count"],
        "correct_count": len(debugger.ground_truth) - analysis["error_analysis"]["error_count"] if debugger.ground_truth is not None else 0,
        "error_indices": error_indices
    }

# New endpoint: Training History
@app.get("/api/training-history", response_model=List[TrainingHistoryItem])
async def get_training_history():
    global debugger, training_history
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    # If history hasn't been generated yet, create some sample data
    if not training_history:
        num_iterations = 10
        training_history = []
        base_accuracy = 0.65
        for i in range(1, num_iterations + 1):
            timestamp = datetime.now().replace(minute=i*5).isoformat()
            training_history.append({
                "iteration": i,
                "accuracy": min(0.98, base_accuracy + (i * 0.03) + random.uniform(-0.01, 0.01)),
                "loss": max(0.05, 0.5 - (i * 0.05) + random.uniform(-0.01, 0.01)),
                "timestamp": timestamp
            })
    
    return training_history

# New endpoint: Error Types
@app.get("/api/error-types", response_model=List[ErrorType])
async def get_error_types():
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    analysis = debugger.analyze()
    error_count = analysis["error_analysis"]["error_count"]
    
    # In a real app, you'd calculate these from the model predictions
    # Here we'll just generate mock data for demonstration
    false_positives = int(error_count * 0.6)
    false_negatives = error_count - false_positives
    
    return [
        {"name": "False Positives", "value": false_positives},
        {"name": "False Negatives", "value": false_negatives}
    ]

# New endpoint: Confusion Matrix
@app.get("/api/confusion-matrix", response_model=ConfusionMatrixResponse)
async def get_confusion_matrix():
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    # This is example data for a binary classification problem
    # In a real app, you'd calculate this from model predictions
    matrix = [
        [45, 5],
        [8, 42]
    ]
    
    labels = ["Negative", "Positive"]
    
    return {"matrix": matrix, "labels": labels}

# New endpoint: Prediction Distribution
@app.get("/api/prediction-distribution", response_model=List[PredictionDistributionItem])
async def get_prediction_distribution():
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    # Example data for a binary classification problem
    # In a real app, you'd calculate this from model predictions
    return [
        {"class_name": "Class 0", "count": 50},
        {"class_name": "Class 1", "count": 50}
    ]

# New endpoint: Sample Predictions
@app.get("/api/sample-predictions")
async def get_sample_predictions(limit: int = 10, offset: int = 0):
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    # Generate mock sample predictions
    # In a real app, you'd return actual model predictions
    samples = []
    for i in range(offset, offset + limit):
        # Random prediction between 0 and 1
        prediction = random.random()
        # True label (0 or 1)
        true_label = 1 if prediction > 0.5 else 0
        # Introduce some errors
        prediction_label = 1 if prediction > 0.5 else 0
        if random.random() < 0.2:  # 20% error rate
            prediction_label = 1 - prediction_label
        
        samples.append({
            "index": i,
            "prediction": prediction,
            "prediction_label": prediction_label,
            "true_label": true_label,
            "is_error": prediction_label != true_label
        })
    
    return {"samples": samples, "total": 100, "limit": limit, "offset": offset}

def start_server(model_debugger, port: int = 8000):
    """Start the FastAPI server with the given ModelDebugger instance."""
    global debugger
    debugger = model_debugger
    
    # Start the server in a blocking way
    uvicorn.run(app, host="0.0.0.0", port=port)
    
    return app