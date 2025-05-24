import asyncio
import logging
import uvicorn
import numpy as np
import time
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re
from typing import List, Dict, Any

import inspect
from pathlib import Path

from fastapi.staticfiles import StaticFiles
import importlib.resources as pkg_resources

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Set matplotlib to use a non-interactive backend
import matplotlib

matplotlib.use("Agg")

app = FastAPI(title="CompileML API")

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

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

# Create a directory for storing visualization images
os.makedirs("temp_visualizations", exist_ok=True)

try:
    from cinder.ml_analysis.code_generator import SimpleCodeGenerator

    HAS_CODE_GENERATOR = True
except ImportError:
    HAS_CODE_GENERATOR = False

# API Models with enhanced documentation
class ModelInfoResponse(BaseModel):
    name: str = Field(..., description="Name of the model")
    framework: str = Field(
        ..., description="ML framework used (pytorch, tensorflow, or sklearn)"
    )
    dataset_size: int = Field(
        ..., description="Number of samples in the evaluation dataset"
    )
    accuracy: float = Field(..., description="Model accuracy on the evaluation dataset")
    precision: Optional[float] = Field(
        None, description="Precision score (weighted average for multi-class)"
    )
    recall: Optional[float] = Field(
        None, description="Recall score (weighted average for multi-class)"
    )
    f1: Optional[float] = Field(
        None, description="F1 score (weighted average for multi-class)"
    )
    roc_auc: Optional[float] = Field(
        None, description="ROC AUC score (for binary classification)"
    )


class ModelCodeResponse(BaseModel):
    code: str = Field(..., description="The model's source code")
    file_path: Optional[str] = Field(None, description="Path to the code file")
    framework: str = Field(..., description="ML framework detected")


class SaveCodeRequest(BaseModel):
    code: str = Field(..., description="Code to save")
    file_path: Optional[str] = Field(None, description="Optional file path to save to")


class ErrorType(BaseModel):
    name: str = Field(..., description="Name of the error type")
    value: int = Field(..., description="Count of errors of this type")
    class_id: Optional[int] = Field(None, description="Class ID for multi-class errors")


class TrainingHistoryItem(BaseModel):
    iteration: int = Field(..., description="Training iteration or epoch number")
    accuracy: float = Field(..., description="Model accuracy at this iteration")
    loss: Optional[float] = Field(None, description="Loss value at this iteration")
    learning_rate: Optional[float] = Field(
        None, description="Learning rate at this iteration"
    )
    timestamp: Optional[str] = Field(
        None, description="Timestamp when this iteration completed"
    )


class PredictionDistributionItem(BaseModel):
    class_name: str = Field(..., description="Class name or ID")
    count: int = Field(..., description="Number of predictions for this class")


class ConfusionMatrixResponse(BaseModel):
    matrix: List[List[int]] = Field(..., description="Confusion matrix values")
    labels: List[str] = Field(
        ..., description="Class labels corresponding to matrix rows/columns"
    )
    num_classes: int = Field(..., description="Number of unique classes")


class ErrorAnalysisResponse(BaseModel):
    error_count: int = Field(..., description="Total number of prediction errors")
    correct_count: int = Field(..., description="Total number of correct predictions")
    error_rate: float = Field(..., description="Error rate (errors/total)")
    error_indices: List[int] = Field(..., description="Indices of samples with errors")
    error_types: Optional[List[Dict[str, Any]]] = Field(
        None, description="Categorized error types"
    )


class ConfidenceAnalysisResponse(BaseModel):
    avg_confidence: float = Field(..., description="Average prediction confidence")
    avg_correct_confidence: float = Field(
        ..., description="Average confidence for correct predictions"
    )
    avg_incorrect_confidence: float = Field(
        ..., description="Average confidence for incorrect predictions"
    )
    calibration_error: float = Field(
        ..., description="Difference between accuracy and average confidence"
    )
    confidence_distribution: Dict[str, Any] = Field(
        ..., description="Distribution of confidence scores"
    )
    overconfident_examples: Dict[str, Any] = Field(
        ..., description="Examples of overconfident predictions"
    )
    underconfident_examples: Dict[str, Any] = Field(
        ..., description="Examples of underconfident predictions"
    )


class FeatureImportanceResponse(BaseModel):
    feature_names: List[str] = Field(..., description="Names of the features")
    importance_values: List[float] = Field(
        ..., description="Importance score for each feature"
    )
    importance_method: str = Field(
        ..., description="Method used to calculate importance"
    )


class CrossValidationResponse(BaseModel):
    fold_results: List[Dict[str, Any]] = Field(
        ..., description="Results for each cross-validation fold"
    )
    mean_accuracy: float = Field(..., description="Mean accuracy across all folds")
    std_accuracy: float = Field(
        ..., description="Standard deviation of accuracy across folds"
    )
    n_folds: int = Field(..., description="Number of cross-validation folds")


class PredictionDriftResponse(BaseModel):
    class_distribution: Dict[str, int] = Field(
        ..., description="Distribution of true classes"
    )
    prediction_distribution: Dict[str, int] = Field(
        ..., description="Distribution of predicted classes"
    )
    drift_scores: Dict[str, float] = Field(
        ..., description="Drift score for each class"
    )
    drifting_classes: List[int] = Field(
        ..., description="Classes with significant drift"
    )
    overall_drift: float = Field(..., description="Overall drift score")


class SamplePrediction(BaseModel):
    index: int = Field(..., description="Sample index")
    prediction: int = Field(..., description="Predicted class")
    true_label: int = Field(..., description="True class label")
    is_error: bool = Field(..., description="Whether the prediction is an error")
    confidence: Optional[float] = Field(
        None, description="Confidence of the prediction"
    )
    probabilities: Optional[List[float]] = Field(
        None, description="Probability for each class"
    )


class SamplePredictionsResponse(BaseModel):
    samples: List[SamplePrediction] = Field(
        ..., description="List of sample predictions"
    )
    total: int = Field(..., description="Total number of samples")
    limit: int = Field(..., description="Maximum number of samples per page")
    offset: int = Field(..., description="Offset for pagination")
    include_errors_only: bool = Field(
        ..., description="Whether only errors are included"
    )


class ROCCurveResponse(BaseModel):
    fpr: List[float] = Field(..., description="False positive rates")
    tpr: List[float] = Field(..., description="True positive rates")
    thresholds: List[float] = Field(..., description="Classification thresholds")


class ServerStatusResponse(BaseModel):
    status: str = Field(..., description="API server status")
    uptime: str = Field(..., description="Server uptime")
    connected_model: Optional[str] = Field(None, description="Name of connected model")
    memory_usage: Optional[float] = Field(None, description="Memory usage in MB")
    version: str = Field("1.0.0", description="API version")
    started_at: str = Field(..., description="Server start time")


class ImprovementSuggestion(BaseModel):
    category: str = Field(..., description="Category of improvement")
    issue: str = Field(..., description="Detected issue")
    suggestion: str = Field(..., description="Suggested improvement")
    severity: float = Field(..., description="How severe the issue is (0-1)")
    impact: float = Field(..., description="Estimated impact of fix (0-1)")
    code_example: str = Field(..., description="Example code for implementation")


# Track server start time
server_start_time = datetime.now()

# Function to clean up old visualization files
def cleanup_old_visualizations(max_age_seconds=3600):  # Default: 1 hour
    """Remove visualization files older than max_age_seconds"""
    current_time = time.time()
    for filename in os.listdir("temp_visualizations"):
        file_path = os.path.join("temp_visualizations", filename)
        if os.path.isfile(file_path):
            # Check file age
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                os.remove(file_path)


# Root endpoint
@app.get("/")
async def root():
    return {"message": "CompileML API is running", "version": "1.0.0"}


# Status endpoint
@app.get("/api/status", response_model=ServerStatusResponse)
async def get_status():
    global debugger, server_start_time

    # Calculate uptime
    uptime = datetime.now() - server_start_time
    uptime_str = str(timedelta(seconds=int(uptime.total_seconds())))

    return {
        "status": "online",
        "uptime": uptime_str,
        "connected_model": debugger.name if debugger else None,
        "memory_usage": np.random.uniform(200, 500),  # Mock memory usage in MB
        "version": "1.0.0",
        "started_at": server_start_time.isoformat(),
    }


@app.get("/api/model-code", response_model=ModelCodeResponse)
async def get_model_code():
    """Get the source code of the current model from the executing script."""
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    try:
        model_code = ""
        file_path = None

        # Method 1: Check if debugger has the source file path stored
        if hasattr(debugger, "source_file_path") and debugger.source_file_path:
            try:
                with open(debugger.source_file_path, "r", encoding="utf-8") as f:
                    model_code = f.read()
                file_path = debugger.source_file_path
                logging.info(f"Loaded code from stored source file: {file_path}")
            except Exception as e:
                logging.warning(
                    f"Could not read stored source file {debugger.source_file_path}: {str(e)}"
                )

        # Method 2: Try to get the main module file (the script that was executed)
        if not model_code:
            try:
                import __main__

                if hasattr(__main__, "__file__") and __main__.__file__:
                    main_file = os.path.abspath(__main__.__file__)
                    with open(main_file, "r", encoding="utf-8") as f:
                        model_code = f.read()
                    file_path = main_file
                    logging.info(f"Loaded code from main module: {file_path}")
            except Exception as e:
                logging.warning(f"Could not read main module file: {str(e)}")

        # Method 3: Try to get source from the calling frame/stack
        if not model_code:
            try:
                import inspect

                # Get the stack and find the first frame that's not from our backend
                for frame_info in inspect.stack():
                    frame_file = frame_info.filename
                    # Skip frames from our backend or system files
                    if (
                        not frame_file.endswith("server.py")
                        and not frame_file.endswith("connector.py")
                        and not "site-packages" in frame_file
                        and not frame_file.startswith("<")
                        and frame_file.endswith(".py")
                    ):

                        with open(frame_file, "r", encoding="utf-8") as f:
                            model_code = f.read()
                        file_path = frame_file
                        logging.info(f"Loaded code from stack frame: {file_path}")
                        break
            except Exception as e:
                logging.warning(f"Could not read from stack frames: {str(e)}")

        # Method 4: Look for common files in current working directory
        if not model_code:
            try:
                current_dir = os.getcwd()
                potential_files = [
                    "run_server.py",
                    "run_2_demo.py",
                    "high_variance.py",
                    "sklearn_demo.py",
                    "tensorflow_demo.py",
                    "model.py",
                    "train.py",
                    "main.py",
                ]

                for filename in potential_files:
                    file_path = os.path.join(current_dir, filename)
                    if os.path.exists(file_path):
                        with open(file_path, "r", encoding="utf-8") as f:
                            model_code = f.read()
                        logging.info(f"Loaded code from common file: {file_path}")
                        break

                    # Also check in examples directory
                    examples_path = os.path.join(current_dir, "examples", filename)
                    if os.path.exists(examples_path):
                        with open(examples_path, "r", encoding="utf-8") as f:
                            model_code = f.read()
                        file_path = examples_path
                        logging.info(f"Loaded code from examples: {file_path}")
                        break

            except Exception as e:
                logging.warning(f"Could not read model file: {str(e)}")

        # Method 5: Generate template if nothing else works
        if not model_code:
            model_code = generate_code_template(debugger.framework)
            file_path = f"generated_template_{debugger.framework.lower()}.py"
            logging.info(f"Generated template for framework: {debugger.framework}")

        return ModelCodeResponse(
            code=model_code, file_path=file_path, framework=debugger.framework
        )

    except Exception as e:
        logging.error(f"Error getting model code: {str(e)}")
        # Return a template as fallback
        return ModelCodeResponse(
            code=generate_code_template(debugger.framework),
            file_path="error_fallback_template.py",
            framework=debugger.framework,
        )


@app.post("/api/model-code")
async def save_model_code(request: SaveCodeRequest):
    """Save the model code to a file."""
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    try:
        # Determine the file path
        if request.file_path:
            file_path = request.file_path
        else:
            # Use a default path based on the current working directory
            current_dir = os.getcwd()
            file_path = os.path.join(current_dir, "saved_model_code.py")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the code
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(request.code)

        logging.info(f"Model code saved to: {file_path}")

        return {
            "message": "Code saved successfully",
            "file_path": file_path,
            "size": len(request.code),
        }

    except Exception as e:
        logging.error(f"Error saving model code: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save code: {str(e)}")


def generate_code_template(framework: str) -> str:
    """Generate a code template based on the ML framework."""

    if framework.lower() == "pytorch":
        return '''import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

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

def generate_synthetic_data(num_samples=500, input_size=10, num_classes=2):
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

def train_model(model, train_loader, num_epochs=10):
    """Train the model with the provided data."""
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
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

# Create and train model
if __name__ == "__main__":
    # Generate data
    X, y = generate_synthetic_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = NeuralNetwork()
    
    # Train model
    train_model(model, dataloader)
'''

    elif framework.lower() == "tensorflow":
        return '''import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def create_model(input_shape, num_classes=2):
    """Create a TensorFlow/Keras model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_synthetic_data(num_samples=500, input_size=10):
    """Generate synthetic data for demonstration."""
    X = np.random.randn(num_samples, input_size)
    weights = np.random.randn(input_size)
    bias = np.random.randn(1)
    scores = np.dot(X, weights) + bias
    y = (scores > 0).astype(int)
    
    # Add some noise
    noise_indices = np.random.choice(num_samples, int(num_samples * 0.1), replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    
    return X, y

def train_model(model, X_train, y_train, X_val, y_val, epochs=20):
    """Train the model."""
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    return history

# Create and train model
if __name__ == "__main__":
    # Generate data
    X, y = generate_synthetic_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model
    model = create_model(X.shape[1])
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_val, y_val)
    print(f'Test Accuracy: {test_accuracy:.4f}')
'''

    else:  # sklearn
        return '''import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def create_model(model_type='random_forest'):
    """Create a scikit-learn model."""
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
    else:
        raise ValueError("Unknown model type")
    
    return model

def generate_synthetic_data(num_samples=500, num_features=10):
    """Generate synthetic data for demonstration."""
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train and evaluate the model."""
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print('\\nClassification Report:')
    print(classification_report(y_test, y_pred))
    
    return accuracy

# Create and train model
if __name__ == "__main__":
    # Generate data
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model
    model = create_model('random_forest')
    
    # Train and evaluate
    accuracy = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
'''


# Model info endpoint
@app.get("/api/model", response_model=ModelInfoResponse)
async def get_model_info():
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    # Get comprehensive model analysis
    analysis = debugger.analyze()

    return {
        "name": debugger.name,
        "framework": debugger.framework,
        "dataset_size": len(debugger.ground_truth)
        if debugger.ground_truth is not None
        else 0,
        "accuracy": analysis["accuracy"],
        "precision": analysis.get("precision"),
        "recall": analysis.get("recall"),
        "f1": analysis.get("f1"),
        "roc_auc": analysis.get("roc_auc"),
    }


@app.get("/api/model-improvements", response_model=Dict[str, Any])
async def get_model_improvements(
    detail_level: str = Query("comprehensive", regex="^(basic|comprehensive|code)$")
):
    """
    Get actionable suggestions to improve model performance.
    """
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    try:
        # Generate improvement suggestions with dynamic code examples
        suggestions = debugger.generate_improvement_suggestions(
            detail_level=detail_level
        )
        return suggestions
    except Exception as e:
        logging.error(f"Error generating improvement suggestions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating suggestions: {str(e)}"
        )


@app.get("/api/generate-code-example", response_model=Dict[str, str])
async def generate_code_example(
    framework: str = Query(..., regex="^(pytorch|tensorflow|sklearn)$"),
    category: str = Query(...),
):
    """
    Generate code example for a specific improvement category and framework.
    """
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    try:
        # Get analysis to provide context
        analysis = debugger.analyze()

        # Create context
        model_context = {
            "accuracy": analysis["accuracy"],
            "error_rate": analysis["error_analysis"]["error_rate"],
            "framework": debugger.framework,
        }

        # Initialize generator
        if not HAS_CODE_GENERATOR:
            return {"code": "# Code generation requires the Gemini API"}

        code_generator = SimpleCodeGenerator()

        # Generate the code
        code = code_generator.generate_code_example(
            framework=framework, category=category, model_context=model_context
        )

        return {"code": code}
    except Exception as e:
        logging.error(f"Error generating code: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating code: {str(e)}")


# Error analysis endpoint
@app.get("/api/errors", response_model=ErrorAnalysisResponse)
async def get_errors():
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    analysis = debugger.analyze()
    error_analysis = analysis["error_analysis"]

    return {
        "error_count": error_analysis["error_count"],
        "correct_count": len(debugger.ground_truth) - error_analysis["error_count"]
        if debugger.ground_truth is not None
        else 0,
        "error_rate": error_analysis["error_rate"],
        "error_indices": error_analysis["error_indices"],
        "error_types": error_analysis.get("error_types"),
    }


# Confidence analysis endpoint
@app.get("/api/confidence-analysis", response_model=ConfidenceAnalysisResponse)
async def get_confidence_analysis():
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    confidence_analysis = debugger.analyze_confidence()

    if "error" in confidence_analysis:
        raise HTTPException(status_code=400, detail=confidence_analysis["error"])

    return confidence_analysis


# Feature importance endpoint
@app.get("/api/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance():
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    importance_analysis = debugger.analyze_feature_importance()

    if "error" in importance_analysis:
        raise HTTPException(status_code=400, detail=importance_analysis["error"])

    return importance_analysis


@app.get("/api/improvement-suggestions", response_model=List[ImprovementSuggestion])
async def get_improvement_suggestions():
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    return debugger.generate_improvement_suggestions()


# Cross-validation endpoint
@app.get("/api/cross-validation", response_model=CrossValidationResponse)
async def get_cross_validation(k_folds: int = Query(5, ge=2, le=10)):
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    cv_results = debugger.perform_cross_validation(k_folds=k_folds)

    if "error" in cv_results:
        raise HTTPException(status_code=400, detail=cv_results["error"])

    return cv_results


# Prediction drift analysis endpoint
@app.get("/api/prediction-drift", response_model=PredictionDriftResponse)
async def get_prediction_drift(threshold: float = Query(0.1, ge=0.01, le=0.5)):
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    drift_analysis = debugger.analyze_prediction_drift(threshold=threshold)

    if "error" in drift_analysis:
        raise HTTPException(status_code=400, detail=drift_analysis["error"])

    return drift_analysis


# ROC curve endpoint (for binary classification)
@app.get("/api/roc-curve", response_model=ROCCurveResponse)
async def get_roc_curve():
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    analysis = debugger.analyze()

    if "roc_curve" not in analysis:
        raise HTTPException(
            status_code=400,
            detail="ROC curve data not available. This may be because the model is not a binary classifier or probability scores are not available.",
        )

    return analysis["roc_curve"]


# Training History endpoint
@app.get("/api/training-history", response_model=List[TrainingHistoryItem])
async def get_training_history():
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    return debugger.get_training_history()


@app.get("/api/model-improvement-suggestions", response_model=Dict[str, Any])
async def get_model_improvement_suggestions(
    detail_level: str = Query("comprehensive", regex="^(basic|comprehensive|code)$")
):
    """
    Get actionable suggestions to improve model performance.

    This endpoint provides specific, targeted suggestions to improve the model,
    based on analyzing its performance, error patterns, and architecture.
    """
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    try:
        suggestions = debugger.get_improvement_suggestions(detail_level=detail_level)
        return suggestions
    except Exception as e:
        logging.error(f"Error generating improvement suggestions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating suggestions: {str(e)}"
        )


# Error Types endpoint
@app.get("/api/error-types", response_model=List[ErrorType])
async def get_error_types():
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    return debugger.analyze_error_types()


# Confusion Matrix endpoint
@app.get("/api/confusion-matrix", response_model=ConfusionMatrixResponse)
async def get_confusion_matrix():
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    analysis = debugger.analyze()
    return analysis["confusion_matrix"]


# Prediction Distribution endpoint
@app.get(
    "/api/prediction-distribution", response_model=List[PredictionDistributionItem]
)
async def get_prediction_distribution():
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    if debugger.predictions is None:
        debugger.analyze()

    # Calculate class distribution in predictions
    unique_classes = np.unique(debugger.predictions)
    distribution = []

    for cls in unique_classes:
        count = np.sum(debugger.predictions == cls)
        distribution.append({"class_name": f"Class {cls}", "count": int(count)})

    return distribution


# Sample Predictions endpoint
@app.get("/api/sample-predictions", response_model=SamplePredictionsResponse)
async def get_sample_predictions(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    errors_only: bool = Query(False),
):
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    return debugger.get_sample_predictions(
        limit=limit, offset=offset, include_errors_only=errors_only
    )


def start_server(model_debugger, port: int = 8000):
    """Start the FastAPI server with the given ModelDebugger instance."""
    global debugger
    debugger = model_debugger

    # Cleanup old visualizations
    cleanup_old_visualizations()

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)

    return app
