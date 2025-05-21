import asyncio
import logging
import uvicorn
import numpy as np
import time
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Set matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

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

# Create a directory for storing visualization images
os.makedirs('temp_visualizations', exist_ok=True)

try:
    from backend.ml_analysis.code_generator import SimpleCodeGenerator
    HAS_CODE_GENERATOR = True
except ImportError:
    HAS_CODE_GENERATOR = False

# API Models with enhanced documentation
class ModelInfoResponse(BaseModel):
    name: str = Field(..., description="Name of the model")
    framework: str = Field(..., description="ML framework used (pytorch, tensorflow, or sklearn)")
    dataset_size: int = Field(..., description="Number of samples in the evaluation dataset")
    accuracy: float = Field(..., description="Model accuracy on the evaluation dataset")
    precision: Optional[float] = Field(None, description="Precision score (weighted average for multi-class)")
    recall: Optional[float] = Field(None, description="Recall score (weighted average for multi-class)")
    f1: Optional[float] = Field(None, description="F1 score (weighted average for multi-class)")
    roc_auc: Optional[float] = Field(None, description="ROC AUC score (for binary classification)")

class ErrorType(BaseModel):
    name: str = Field(..., description="Name of the error type")
    value: int = Field(..., description="Count of errors of this type")
    class_id: Optional[int] = Field(None, description="Class ID for multi-class errors")

class TrainingHistoryItem(BaseModel):
    iteration: int = Field(..., description="Training iteration or epoch number")
    accuracy: float = Field(..., description="Model accuracy at this iteration")
    loss: Optional[float] = Field(None, description="Loss value at this iteration")
    learning_rate: Optional[float] = Field(None, description="Learning rate at this iteration")
    timestamp: Optional[str] = Field(None, description="Timestamp when this iteration completed")

class PredictionDistributionItem(BaseModel):
    class_name: str = Field(..., description="Class name or ID")
    count: int = Field(..., description="Number of predictions for this class")

class ConfusionMatrixResponse(BaseModel):
    matrix: List[List[int]] = Field(..., description="Confusion matrix values")
    labels: List[str] = Field(..., description="Class labels corresponding to matrix rows/columns")
    num_classes: int = Field(..., description="Number of unique classes")

class ErrorAnalysisResponse(BaseModel):
    error_count: int = Field(..., description="Total number of prediction errors")
    correct_count: int = Field(..., description="Total number of correct predictions")
    error_rate: float = Field(..., description="Error rate (errors/total)")
    error_indices: List[int] = Field(..., description="Indices of samples with errors")
    error_types: Optional[List[Dict[str, Any]]] = Field(None, description="Categorized error types")

class ConfidenceAnalysisResponse(BaseModel):
    avg_confidence: float = Field(..., description="Average prediction confidence")
    avg_correct_confidence: float = Field(..., description="Average confidence for correct predictions")
    avg_incorrect_confidence: float = Field(..., description="Average confidence for incorrect predictions")
    calibration_error: float = Field(..., description="Difference between accuracy and average confidence")
    confidence_distribution: Dict[str, Any] = Field(..., description="Distribution of confidence scores")
    overconfident_examples: Dict[str, Any] = Field(..., description="Examples of overconfident predictions")
    underconfident_examples: Dict[str, Any] = Field(..., description="Examples of underconfident predictions")

class FeatureImportanceResponse(BaseModel):
    feature_names: List[str] = Field(..., description="Names of the features")
    importance_values: List[float] = Field(..., description="Importance score for each feature")
    importance_method: str = Field(..., description="Method used to calculate importance")

class CrossValidationResponse(BaseModel):
    fold_results: List[Dict[str, Any]] = Field(..., description="Results for each cross-validation fold")
    mean_accuracy: float = Field(..., description="Mean accuracy across all folds")
    std_accuracy: float = Field(..., description="Standard deviation of accuracy across folds")
    n_folds: int = Field(..., description="Number of cross-validation folds")

class PredictionDriftResponse(BaseModel):
    class_distribution: Dict[str, int] = Field(..., description="Distribution of true classes")
    prediction_distribution: Dict[str, int] = Field(..., description="Distribution of predicted classes")
    drift_scores: Dict[str, float] = Field(..., description="Drift score for each class")
    drifting_classes: List[int] = Field(..., description="Classes with significant drift")
    overall_drift: float = Field(..., description="Overall drift score")

class SamplePrediction(BaseModel):
    index: int = Field(..., description="Sample index")
    prediction: int = Field(..., description="Predicted class")
    true_label: int = Field(..., description="True class label")
    is_error: bool = Field(..., description="Whether the prediction is an error")
    confidence: Optional[float] = Field(None, description="Confidence of the prediction")
    probabilities: Optional[List[float]] = Field(None, description="Probability for each class")

class SamplePredictionsResponse(BaseModel):
    samples: List[SamplePrediction] = Field(..., description="List of sample predictions")
    total: int = Field(..., description="Total number of samples")
    limit: int = Field(..., description="Maximum number of samples per page")
    offset: int = Field(..., description="Offset for pagination")
    include_errors_only: bool = Field(..., description="Whether only errors are included")

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
    for filename in os.listdir('temp_visualizations'):
        file_path = os.path.join('temp_visualizations', filename)
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
        "started_at": server_start_time.isoformat()
    }

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
        "dataset_size": len(debugger.ground_truth) if debugger.ground_truth is not None else 0,
        "accuracy": analysis["accuracy"],
        "precision": analysis.get("precision"),
        "recall": analysis.get("recall"),
        "f1": analysis.get("f1"),
        "roc_auc": analysis.get("roc_auc")
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
        suggestions = debugger.generate_improvement_suggestions(detail_level=detail_level)
        return suggestions
    except Exception as e:
        logging.error(f"Error generating improvement suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating suggestions: {str(e)}")

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
            "framework": debugger.framework
        }
        
        # Initialize generator
        if not HAS_CODE_GENERATOR:
            return {"code": "# Code generation requires the Gemini API"}
            
        code_generator = SimpleCodeGenerator()
        
        # Generate the code
        code = code_generator.generate_code_example(
            framework=framework,
            category=category,
            model_context=model_context
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
        "correct_count": len(debugger.ground_truth) - error_analysis["error_count"] if debugger.ground_truth is not None else 0,
        "error_rate": error_analysis["error_rate"],
        "error_indices": error_analysis["error_indices"],
        "error_types": error_analysis.get("error_types")
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
        raise HTTPException(status_code=400, detail="ROC curve data not available. This may be because the model is not a binary classifier or probability scores are not available.")
        
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
        raise HTTPException(status_code=500, detail=f"Error generating suggestions: {str(e)}")
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
@app.get("/api/prediction-distribution", response_model=List[PredictionDistributionItem])
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
        distribution.append({
            "class_name": f"Class {cls}",
            "count": int(count)
        })
    
    return distribution

# Sample Predictions endpoint
@app.get("/api/sample-predictions", response_model=SamplePredictionsResponse)
async def get_sample_predictions(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    errors_only: bool = Query(False)
):
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    return debugger.get_sample_predictions(
        limit=limit,
        offset=offset,
        include_errors_only=errors_only
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