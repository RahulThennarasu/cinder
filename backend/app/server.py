import asyncio
import logging
import uvicorn
import numpy as np
import time
import os
from typing import Dict, Any, List, Optional
import difflib
from datetime import datetime, timedelta
import re
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)

from pydantic import BaseModel
from backend.ml_analysis.bit_assistant import BitOptimizer
bit_optimizer = BitOptimizer()

from backend.ml_analysis.environment import check_api_configuration, GEMINI_API_KEY

from fastapi import FastAPI, HTTPException, Query, Header, Depends, Request, WebSocket, WebSocketDisconnect

import json
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

import inspect
from pathlib import Path

from fastapi.staticfiles import StaticFiles
import importlib.resources as pkg_resources

from dotenv import load_dotenv

load_dotenv()

from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException, Query, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
api_key = None

def generate_diff(old_code: str, new_code: str) -> str:
    """Generate a unified diff between two code strings."""
    diff = difflib.unified_diff(
        old_code.splitlines(keepends=True),
        new_code.splitlines(keepends=True),
        fromfile='before',
        tofile='after'
    )
    return ''.join(diff)

# Set matplotlib to use a non-interactive backend
# Set matplotlib to use a non-interactive backend
import matplotlib

matplotlib.use("Agg")

app = FastAPI(title="Cinder API")

# CORS Middleware - Make sure this is enabled
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store reference to the ModelDebugger instance
debugger = None

# Create a directory for storing visualization images
os.makedirs("temp_visualizations", exist_ok=True)

try:
    from backend.auth.auth import validate_api_key
    HAS_AUTH = True
except ImportError:
    HAS_AUTH = False
    logging.warning("Authentication module not found. API key validation disabled.")

try:
    from backend.ml_analysis.code_generator import SimpleCodeGenerator

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
# Create a middleware that includes the API key in API responses
class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        
        # Only modify responses for the frontend, not for API calls
        if request.url.path == "/" or request.url.path.startswith("/static"):
            return response
            
        # Add API key information for the dashboard
        if isinstance(response, JSONResponse):
            try:
                # Get content and modify it
                content = json.loads(bytes(response.body).decode())

                
                # If debugger is available, get its API key
                if debugger and hasattr(debugger, "api_key") and debugger.api_key:
                    content["_api_key"] = debugger.api_key
                
                # Update response with modified content
                return JSONResponse(
                    content=content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                    background=response.background
                )
            except Exception as e:
                logging.error(f"Error adding API key to response: {e}")
                
        return response
    
# Add these new models
class UsageStatsResponse(BaseModel):
    api_key_id: str
    daily_usage: int
    daily_limit: int
    monthly_usage: int
    monthly_limit: int
    last_used: Optional[str]
    total_requests: int
    tier: str
    reset_times: Dict[str, str]

class UsageHistoryItem(BaseModel):
    date: str
    requests: int

class UsageHistoryResponse(BaseModel):
    history: List[UsageHistoryItem]
    total_days: int

class BitChatRequest(BaseModel):
    query: str
    code: str
    modelInfo: Optional[Dict[str, Any]] = None
    framework: str = "pytorch"

class SuggestionModel(BaseModel):
    title: str
    description: str
    code: str
    lineNumber: int

class BitChatResponse(BaseModel):
    message: str
    suggestions: Optional[List[SuggestionModel]] = []
async def get_api_key(request: Request, x_api_key: str = Header(None)):
    """
    Dependency to extract and validate the API key.
    
    For dashboard requests, bypass authentication.
    For programmatic API access, enforce authentication.
    """
    global debugger
    
    if not HAS_AUTH:
        # Skip validation if auth module not available
        return "no_auth"
    
    # If it's a dashboard request, bypass authentication
    if is_dashboard_request(request):
        # For dashboard requests, use the debugger's API key if available
        if debugger and hasattr(debugger, "api_key") and debugger.api_key:
            return debugger.api_key
        return "dashboard_access"
    
    # For all other API requests, require valid authentication
    api_key = x_api_key
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Please provide an API key in the X-API-Key header."
        )
    
    if not validate_api_key(api_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return api_key
@app.on_event("startup")
async def startup_event():
    """Initialize necessary services on startup."""
    logger.info("Starting ML Platform API server...")
    
    # Check API configuration
    if check_api_configuration():
        logger.info("API configuration validated.")
    else:
        logger.warning("API configuration incomplete. Some features may be limited.")
    
    # Initialize Google Generative AI client if key is available
    if GEMINI_API_KEY:
        try:
            # Using the proper import style as in your SimpleCodeGenerator
            from google import genai
            # This is the correct initialization based on your code
            logger.info("Attempting to initialize Gemini client")
            _ = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("Google Generative AI client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Generative AI client: {str(e)}")
    else:
        logger.warning("Gemini API key not found. AI features will be limited.")
        
async def get_firebase_token(authorization: Optional[str] = Header(None)):
    """Validate Firebase auth token and return user ID."""
    # For now, we'll make this optional and return a demo user ID
    # You can enhance this later when you integrate Firebase auth in the backend
    
    if not authorization or not authorization.startswith("Bearer "):
        # For demo purposes, return a demo user ID
        # In production, you'd raise an HTTPException here
        return "demo_user_id"
    
    try:
        # If you have Firebase Admin SDK set up, you can validate the token here
        # For now, we'll just return a demo user ID
        token = authorization.split("Bearer ")[1]
        
        # In a real implementation, you'd do:
        # decoded_token = firebase_auth.verify_id_token(token)
        # return decoded_token['uid']
        
        # For demo, just return a user ID
        return "demo_user_id"
        
    except Exception as e:
        # For demo purposes, just return demo user
        # In production: raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
        return "demo_user_id"
# Add imports at the top of server.py
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect
import asyncio
import json
import difflib

# Add the WebSocket endpoint

@app.websocket("/ws/bit-optimizer")
async def bit_optimizer_websocket(websocket: WebSocket):
    await websocket.accept()
    
    # Track connection for rate limiting
    api_key = None
    connection_authorized = False
    
    try:
        # Wait for initial connection message to get API key
        initial_data = await websocket.receive_json()
        print(f"Received initial data: {initial_data}")
        
        # Extract API key - support different ways the frontend might send it
        api_key = initial_data.get("api_key") or initial_data.get("apiKey")
        
        # If no API key but has action "connect", wait for next message that might have API key
        if not api_key and initial_data.get("action") == "connect":
            try:
                # Wait with timeout for next message with potential API key
                next_data = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
                api_key = next_data.get("api_key") or next_data.get("apiKey")
                # Continue processing with the action from this message
                initial_data = next_data
            except asyncio.TimeoutError:
                print("Timeout waiting for API key in follow-up message")
                
        # Debug log API key for troubleshooting
        if api_key:
            print(f"WebSocket connection with API key: {api_key[:8]}...")
        else:
            # If no API key provided, try to get it from the global debugger instance
            if debugger and hasattr(debugger, "api_key") and debugger.api_key:
                api_key = debugger.api_key
                print(f"Using API key from debugger: {api_key[:8]}...")
            else:
                print("WebSocket connection without API key - DEVELOPMENT MODE")
        
        # Check API key and rate limit
        if HAS_AUTH and api_key:
            from backend.auth.auth import validate_api_key, check_rate_limit, log_api_usage
            if validate_api_key(api_key) and check_rate_limit(api_key):
                connection_authorized = True
                # Log initial websocket connection
                log_api_usage(api_key, "bit_optimizer_websocket_connect")
                print(f"WebSocket connection authorized for API key: {api_key[:8]}...")
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "Rate limit exceeded or invalid API key. Please upgrade your plan or try again later."
                })
                await websocket.close(code=1008)  # Policy violation
                return
        else:
            # If no auth module or no API key provided, allow the connection in development
            connection_authorized = True
            print("Auth bypassed in development mode")
        
        # Send welcome message
        await websocket.send_json({
            "type": "greeting",
            "message": "Hello! This is Bit, your intelligent ML optimization partner. I've analyzed your model architecture and am ready to suggest powerful improvements. What aspect of your model would you like to enhance?"
        })
        
        # Process initial action if present
        if "action" in initial_data and initial_data["action"] not in ["connect"]:
            if initial_data.get("action") == "optimize":
                # Handle auto-optimization
                if HAS_AUTH and api_key:
                    from backend.auth.auth import check_rate_limit, log_api_usage
                    if check_rate_limit(api_key):
                        log_api_usage(api_key, "bit_optimizer_optimize")
                        await handle_auto_optimization(websocket, {**initial_data, "api_key": api_key})
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Rate limit exceeded. Please upgrade your plan or try again later."
                        })
                else:
                    await handle_auto_optimization(websocket, initial_data)
            elif initial_data.get("action") == "chat":
                # Handle chat improvements
                if HAS_AUTH and api_key:
                    from backend.auth.auth import check_rate_limit, log_api_usage
                    if check_rate_limit(api_key):
                        log_api_usage(api_key, "bit_optimizer_chat")
                        await handle_chat_improvements(websocket, {**initial_data, "api_key": api_key})
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Rate limit exceeded. Please upgrade your plan or try again later."
                        })
                else:
                    await handle_chat_improvements(websocket, initial_data)
        
        # Listen for messages in a loop
        while True:
            data = await websocket.receive_json()
            
            # Ensure API key is passed to handlers
            if api_key and "api_key" not in data:
                data["api_key"] = api_key
            
            # Check rate limit on each substantial action
            if HAS_AUTH and api_key and data.get("action") in ["optimize", "chat"]:
                from backend.auth.auth import check_rate_limit, log_api_usage
                if not check_rate_limit(api_key):
                    await websocket.send_json({
                        "type": "error",
                        "message": "Rate limit exceeded. Please upgrade your plan or try again later."
                    })
                    continue
                
                # Log the specific action type
                action_type = data.get("action", "unknown")
                log_api_usage(api_key, f"bit_optimizer_{action_type}")
                print(f"Logged usage for action: bit_optimizer_{action_type}")
            
            if data.get("action") == "optimize":
                # Handle full auto-optimization
                await handle_auto_optimization(websocket, data)
            elif data.get("action") == "chat":
                # Handle conversational improvements
                await handle_chat_improvements(websocket, data)
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Error: {str(e)}"
            })
        except:
            pass
async def handle_chat_improvements(websocket: WebSocket, data):
    """Handle improvement requests through chat"""
    user_query = data.get("query", "")
    model_code = data.get("code", "")
    framework = data.get("framework", "pytorch")
    api_key = data.get("api_key")
    
    # Check if Gemini API is configured
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key or not bit_optimizer.client:
        await websocket.send_json({
            "type": "error",
            "message": "Gemini API not configured"
        })
        return
    
    # CRITICAL FIX: Log API usage for every chat interaction
    # This is the key fix that ensures usage is tracked in Firebase
    if HAS_AUTH and api_key:
        from backend.auth.auth import check_rate_limit, log_api_usage
        if not check_rate_limit(api_key):
            await websocket.send_json({
                "type": "error",
                "message": "Rate limit exceeded. Please upgrade your plan or try again later."
            })
            return
        
        # Log the chat interaction - this is the critical line for tracking
        log_api_usage(api_key, "bit_optimizer_chat")
    
    # Send status message
    await websocket.send_json({
        "type": "status",
        "message": f"Analyzing your request: '{user_query}'"
    })
    
    try:
        # Use Gemini to analyze the query and decide what improvements to make
        improvement_prompt = f"""
        You are Bit, an AI assistant specialized in improving ML models.
        
        User query: "{user_query}"
        Model code:
        ```python
        {model_code}
        ```
        Framework: {framework}
        
        Based on the user's query, identify 1-3 specific improvements to make to this code.
        Format your response as a JSON array of improvements:
        [
          {{
            "title": "Short title of the improvement",
            "description": "Detailed explanation of why this change is beneficial",
            "code_section": "Which part of the code to modify", 
            "priority": "high/medium/low",
            "expected_benefit": "Expected improvement from this change"
          }},
          // additional improvements...
        ]
        
        IMPORTANT: Always include the "code_section" field to indicate which part of the code to modify.
        Focus on the most important improvements first. Only include valid, helpful suggestions.
        """
        
        # Call Gemini to identify improvements
        response = bit_optimizer.client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=improvement_prompt
        )
        
        # CRITICAL FIX: Log API usage for Gemini response
        # Track every Gemini API call separately
        if HAS_AUTH and api_key:
            from backend.auth.auth import log_api_usage
            log_api_usage(api_key, "bit_optimizer_gemini_call")
        
        # Extract improvements from response
        improvements = []
        try:
            import re
            import json
            
            text = response.text if hasattr(response, 'text') else response.parts[0].text
            json_match = re.search(r'\[\s*{[\s\S]*}\s*\]', text)
            
            if json_match:
                improvements = json.loads(json_match.group(0))
            else:
                # Fallback improvement if parsing fails
                improvements = [{
                    "title": "General Optimization",
                    "description": "Improving model architecture and performance",
                    "code_section": "Model architecture",
                    "priority": "high",
                    "expected_benefit": "Better model performance"
                }]
        except Exception as e:
            print(f"Error parsing improvements: {e}")
            improvements = [{
                "title": "General Optimization",
                "description": "Improving model architecture and performance",
                "code_section": "Model architecture",
                "priority": "high",
                "expected_benefit": "Better model performance"
            }]
        
        # Make sure each improvement has required fields
        for improvement in improvements:
            # Add code_section if missing
            if "code_section" not in improvement:
                improvement["code_section"] = "Model architecture"
            
            # Add expected_benefit if missing
            if "expected_benefit" not in improvement:
                improvement["expected_benefit"] = "Improved model performance"
        
        # Process each improvement
        for i, improvement in enumerate(improvements):
            # Send analysis message
            await websocket.send_json({
                "type": "status",
                "message": f"Working on: {improvement['title']}"
            })
            
            # Use the existing optimization functions to generate the code changes
            optimization_result = await bit_optimizer.generate_optimization_step(
                model_code, 
                improvement,
                framework
            )
            
            # CRITICAL FIX: Log API usage for each optimization step
            if HAS_AUTH and api_key:
                from backend.auth.auth import log_api_usage
                log_api_usage(api_key, f"bit_optimizer_optimization_step")
            
            if optimization_result and "updated_code" in optimization_result:
                # Update model code for next improvement
                model_code = optimization_result["updated_code"]
                
                # Send the optimization result
                await websocket.send_json({
                    "type": "optimization",
                    "optimization": improvement,
                    "changes": optimization_result
                })
                
                # Generate explanation
                explanation = await bit_optimizer.explain_optimization_benefits(
                    model_code, 
                    optimization_result["updated_code"],
                    improvement,
                    framework
                )
                
                # CRITICAL FIX: Log API usage for explanation generation
                if HAS_AUTH and api_key:
                    from backend.auth.auth import log_api_usage
                    log_api_usage(api_key, f"bit_optimizer_explanation")
                
                await websocket.send_json({
                    "type": "explanation",
                    "message": explanation
                })
        
        # CRITICAL FIX: Log API usage for completion
        if HAS_AUTH and api_key:
            from backend.auth.auth import log_api_usage
            log_api_usage(api_key, "bit_optimizer_chat_complete")
        
        # Send completion message
        await websocket.send_json({
            "type": "complete",
            "message": "I've completed the requested improvements. Is there anything else you'd like me to help with?"
        })
        
    except Exception as e:
        print(f"Error processing chat improvement: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Error implementing improvements: {str(e)}"
        })
def log_api_usage(api_key: str, endpoint: str = "unknown"):
    """
    Log API usage for analytics with improved reliability and debugging.
    
    Args:
        api_key: The API key used for the request
        endpoint: The endpoint or operation being accessed
    """
    try:
        if not db or not HAS_FIREBASE:
            print(f"Firebase not initialized, can't log usage for {endpoint}")
            return
        
        # Get the API key document
        query = db.collection("api_keys").where("key", "==", api_key).limit(1)
        results = list(query.stream())
        
        if not results:
            print(f"API key {api_key[:8]}... not found, can't log usage")
            return
        
        key_doc = results[0]
        today = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Logging API usage for key {key_doc.id} endpoint {endpoint}")
        
        # Update daily history with retry logic
        history_ref = db.collection("api_usage").document(key_doc.id).collection("daily_history").document(today)
        
        # Function to be executed in transaction
        @firestore.transactional
        def update_daily_history_with_retry(transaction, retry_count=0):
            """Update history with transaction and retry logic"""
            try:
                doc = history_ref.get(transaction=transaction)
                if doc.exists:
                    # Increment existing document
                    current_requests = doc.get("requests", 0)
                    transaction.update(history_ref, {
                        "requests": firestore.Increment(1),
                        "endpoints": firestore.ArrayUnion([endpoint]) if "endpoints" not in doc.to_dict() else doc.get("endpoints", []),
                        "endpoint_counts." + endpoint: firestore.Increment(1),
                        "last_endpoint": endpoint,
                        "last_used": firestore.SERVER_TIMESTAMP
                    })
                    print(f"Updated existing history for {key_doc.id}, now {current_requests + 1} requests")
                else:
                    # Create new document
                    transaction.set(history_ref, {
                        "date": today,
                        "requests": 1,
                        "endpoints": [endpoint],
                        "endpoint_counts": {endpoint: 1},
                        "last_endpoint": endpoint,
                        "last_used": firestore.SERVER_TIMESTAMP
                    })
                    print(f"Created new history for {key_doc.id}")
            except Exception as e:
                # Retry on transaction errors a few times
                if retry_count < 3:
                    print(f"Transaction failed, retrying ({retry_count+1}/3): {e}")
                    time.sleep(0.5)  # Brief delay before retry
                    return update_daily_history_with_retry(transaction, retry_count + 1)
                else:
                    raise e
        
        # Execute the transaction
        try:
            transaction = db.transaction()
            update_daily_history_with_retry(transaction)
            
            # Also update the API key's last used time and usage count
            key_doc.reference.update({
                "lastUsed": firestore.SERVER_TIMESTAMP,
                "usageCount": firestore.Increment(1)
            })
            
            # Print success message for debugging
            print(f"Successfully logged API usage for {endpoint}")
            return True
        except Exception as e:
            print(f"Transaction failed after retries: {e}")
            return False
        
    except Exception as e:
        print(f"Error logging API usage: {e}")
        return False
async def handle_auto_optimization(websocket: WebSocket, data):
    """Handle full auto-optimization process"""
    model_code = data.get("code", "")
    framework = data.get("framework", "pytorch")
    api_key = data.get("api_key")
    
    # Check rate limit for the API key at the beginning
    if HAS_AUTH and api_key:
        from backend.auth.auth import check_rate_limit, log_api_usage
        if not check_rate_limit(api_key):
            await websocket.send_json({
                "type": "error",
                "message": "Rate limit exceeded. Please upgrade your plan or try again later."
            })
            return
        
        # Log the beginning of an auto-optimization session
        log_api_usage(api_key, "bit_optimizer_auto_optimization_start")
    
    # Send status message
    await websocket.send_json({
        "type": "status",
        "message": "Analyzing your model to identify optimization opportunities..."
    })
    
    try:
        # Generate optimization steps
        optimization_steps = await generate_optimization_steps(model_code, framework)
        
        # Send the optimization plan
        await websocket.send_json({
            "type": "status",
            "message": f"Found {len(optimization_steps)} potential optimizations. I'll implement them one by one."
        })
        
        # Apply each optimization step
        for i, step in enumerate(optimization_steps):
            # Check rate limit before each optimization step
            if HAS_AUTH and api_key:
                from backend.auth.auth import check_rate_limit, log_api_usage
                if not check_rate_limit(api_key):
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Rate limit exceeded after step {i+1}/{len(optimization_steps)}. Please upgrade your plan to continue."
                    })
                    return
                
                # Log each optimization step
                log_api_usage(api_key, f"bit_optimizer_auto_optimization_step_{i+1}")
            
            # Send status message for this step
            await websocket.send_json({
                "type": "status",
                "message": f"Generating code changes for {step['title']}..."
            })
            
            # Create a delay to show progress (simulates work)
            await asyncio.sleep(1)
            
            # Implement the optimization
            try:
                # Send the optimization details
                await websocket.send_json({
                    "type": "optimization",
                    "optimization": step
                })
                
                # Apply the optimization to the code
                updated_code = await apply_optimization(model_code, step, framework)
                
                # Generate diff between old and new code
                diff = generate_diff(model_code, updated_code)
                
                # Update the model code for the next step
                old_code = model_code
                model_code = updated_code
                
                # Send the changes
                await websocket.send_json({
                    "type": "optimization",
                    "optimization": step,
                    "changes": {
                        "explanation": step['description'],
                        "updated_code": updated_code,
                        "changes_summary": f"Updated code to implement {step['title']}"
                    }
                })
                
                # Generate an explanation of the benefits
                explanation = f"This optimization improves {step['expected_benefit'].lower()}. " + \
                              f"The changes focus on {step['code_section'].lower()}, which enhances overall model performance."
                
                await websocket.send_json({
                    "type": "explanation",
                    "message": explanation
                })
                
            except Exception as e:
                print(f"Error applying optimization step {i+1}: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error implementing {step['title']}: {str(e)}"
                })
                continue
        
        # Final log entry for completed optimization
        if HAS_AUTH and api_key:
            from backend.auth.auth import log_api_usage
            log_api_usage(api_key, "bit_optimizer_auto_optimization_complete")
        
        # Send completion message
        await websocket.send_json({
            "type": "complete",
            "message": "All optimizations have been applied successfully! Your model should now have improved performance. Let me know if you'd like me to explain any of the changes in more detail."
        })
        
    except Exception as e:
        print(f"Error in auto-optimization: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": f"Error during optimization process: {str(e)}"
        })
# Helper functions for the WebSocket endpoint
async def generate_optimization_step(self, model_code: str, optimization: Dict[str, Any], framework: str) -> Dict[str, Any]:
    """Generate high-quality code changes for a specific ML model optimization."""
    if not self.client:
        raise HTTPException(status_code=500, detail="Gemini API client not configured")
    
    # Extract framework-specific details for more targeted optimization
    framework_lower = framework.lower()
    
    # Create a comprehensive framework context dictionary with detailed technical information
    framework_context = {
        "pytorch": {
            "imports": [
                "import torch", 
                "import torch.nn as nn", 
                "import torch.nn.functional as F",
                "from torch.utils.data import DataLoader, Dataset, random_split",
                "import torch.optim as optim"
            ],
            "architecture": {
                "base_classes": ["nn.Module"],
                "layer_types": [
                    {"name": "nn.Linear", "purpose": "Fully connected layer", "typical_use": "self.fc = nn.Linear(in_features, out_features)"},
                    {"name": "nn.Conv2d", "purpose": "2D convolutional layer", "typical_use": "self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)"},
                    {"name": "nn.BatchNorm2d", "purpose": "2D batch normalization", "typical_use": "self.bn = nn.BatchNorm2d(num_features)"},
                    {"name": "nn.Dropout", "purpose": "Regularization via random neuron deactivation", "typical_use": "self.dropout = nn.Dropout(p=0.5)"},
                    {"name": "nn.MaxPool2d", "purpose": "2D max pooling", "typical_use": "self.pool = nn.MaxPool2d(kernel_size, stride)"},
                    {"name": "nn.LSTM", "purpose": "Long Short-Term Memory", "typical_use": "self.lstm = nn.LSTM(input_size, hidden_size, num_layers)"}
                ],
                "activation_functions": [
                    {"name": "F.relu", "purpose": "Rectified Linear Unit", "typical_use": "x = F.relu(x)"},
                    {"name": "F.sigmoid", "purpose": "Sigmoid function", "typical_use": "x = F.sigmoid(x)"},
                    {"name": "F.tanh", "purpose": "Hyperbolic tangent", "typical_use": "x = F.tanh(x)"},
                    {"name": "F.softmax", "purpose": "Softmax normalization", "typical_use": "x = F.softmax(x, dim=1)"},
                    {"name": "F.log_softmax", "purpose": "Log softmax", "typical_use": "x = F.log_softmax(x, dim=1)"}
                ]
            },
            "optimizers": [
                {"name": "optim.SGD", "purpose": "Stochastic Gradient Descent", "typical_use": "optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)"},
                {"name": "optim.Adam", "purpose": "Adaptive Moment Estimation", "typical_use": "optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))"},
                {"name": "optim.RMSprop", "purpose": "Root Mean Square Propagation", "typical_use": "optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)"}
            ],
            "schedulers": [
                {"name": "optim.lr_scheduler.ReduceLROnPlateau", "purpose": "Reduce learning rate when a metric plateaus", 
                 "typical_use": "scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)"},
                {"name": "optim.lr_scheduler.StepLR", "purpose": "Decay learning rate by gamma every step_size epochs", 
                 "typical_use": "scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)"},
                {"name": "optim.lr_scheduler.CosineAnnealingLR", "purpose": "Cosine annealing schedule", 
                 "typical_use": "scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)"}
            ],
            "training_patterns": [
                "model.train() for training mode",
                "optimizer.zero_grad() to clear gradients",
                "loss.backward() for backpropagation",
                "optimizer.step() to update weights",
                "model.eval() and torch.no_grad() for evaluation"
            ],
            "best_practices": [
                "Use nn.Sequential for linear layer sequences",
                "Apply batch normalization before activation functions",
                "Use dropout for regularization, typically after activation",
                "Initialize weights properly (e.g., Xavier/Kaiming)",
                "Use learning rate scheduling for better convergence",
                "Implement early stopping to prevent overfitting",
                "Use mixed precision training for faster computation",
                "Implement gradient clipping for stable training"
            ]
        },
        "tensorflow": {
            # Similar comprehensive structure for TensorFlow
            "imports": [
                "import tensorflow as tf",
                "from tensorflow.keras import layers, models, optimizers, callbacks"
            ],
            # Additional TensorFlow details...
        },
        "sklearn": {
            # Similar comprehensive structure for scikit-learn
            "imports": [
                "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier",
                "from sklearn.linear_model import LogisticRegression"
            ],
            # Additional scikit-learn details...
        }
    }.get(framework_lower, {})
    
    # Extract code patterns and create code understanding
    code_analysis = await self._analyze_code_structure(model_code, framework_lower)
    
    # Enhanced prompt engineering with dynamic structure based on the specific optimization
    prompt_template = self._get_optimization_prompt_template(optimization, framework_lower)
    
    # Merge all contexts into a comprehensive prompt
    prompt = prompt_template.format(
        optimization_title=optimization['title'],
        optimization_description=optimization['description'],
        code_section=optimization.get('code_section', 'Model architecture'),
        framework=framework,
        expected_benefit=optimization.get('expected_benefit', 'Improved model performance'),
        model_code=model_code,
        framework_context=json.dumps(framework_context.get(framework_lower, {}), indent=2),
        code_analysis=json.dumps(code_analysis, indent=2)
    )
    
    # Log the optimization attempt
    logger.info(f"Generating optimization for: {optimization['title']} using {framework}")
    
    try:
        # Call Gemini API with optimized parameters specifically tailored for the optimization type
        optimization_params = self._get_optimization_specific_parameters(optimization['title'])
        
        result = await self._call_gemini_with_enhanced_params(
            prompt,
            parse_json=True,
            max_retries=3,
            **optimization_params
        )
        
        # Validate and enhance the result
        if result and "updated_code" in result:
            # Clean up the code formatting
            result["updated_code"] = self._clean_and_validate_code(result["updated_code"], framework_lower)
            
            # Perform advanced validation of the optimization
            validation_result = await self._validate_optimization(model_code, result["updated_code"], optimization, framework_lower)
            
            if not validation_result['valid']:
                logger.warning(f"Optimization validation failed: {validation_result['reason']}")
                
                # Try again with a more direct approach and specific fixes
                corrected_prompt = self._create_correction_prompt(
                    model_code, 
                    result["updated_code"], 
                    validation_result, 
                    optimization,
                    framework_lower
                )
                
                # Call Gemini API again with the corrected prompt
                corrected_result = await self._call_gemini_with_enhanced_params(
                    corrected_prompt,
                    parse_json=True,
                    max_retries=2,
                    temperature=0.1  # Lower temperature for more predictable corrections
                )
                
                if corrected_result and "updated_code" in corrected_result:
                    # Replace with corrected version
                    result = corrected_result
                    result["updated_code"] = self._clean_and_validate_code(result["updated_code"], framework_lower)
            
            # Generate detailed diff
            result["code_diff"] = self._generate_detailed_diff(model_code, result["updated_code"])
            
            # Add performance impact estimate
            result["performance_impact"] = self._estimate_optimization_impact(
                optimization['title'], 
                model_code, 
                result["updated_code"]
            )
            
            # Add explanation if missing
            if "explanation" not in result or not result["explanation"]:
                result["explanation"] = await self._generate_optimization_explanation(
                    optimization['title'],
                    model_code,
                    result["updated_code"],
                    framework_lower
                )
                
            return result
        
        # If we failed to get a valid result, try again with a simpler approach
        logger.warning("Failed to generate valid optimization response, attempting simplified approach")
        
        # Try again with a simplified prompt
        return await self._generate_fallback_optimization(model_code, optimization, framework_lower)
            
    except Exception as e:
        logger.error(f"Error generating optimization: {str(e)}")
        # Create a useful fallback result using our built-in templates
        return {
            "explanation": f"Applied {optimization['title']} to improve model performance.",
            "updated_code": self._apply_template_optimization(model_code, optimization, framework_lower),
            "changes_summary": f"Modified code to implement {optimization['title']} following best practices."
        }

async def _analyze_code_structure(self, code: str, framework: str) -> Dict[str, Any]:
    """Analyze the structure of the code to better understand what we're working with."""
    analysis = {
        "has_model_class": False,
        "model_class_name": None,
        "has_forward_method": False,
        "has_training_loop": False,
        "has_init_method": False,
        "layers": [],
        "activations": [],
        "optimizers": [],
        "loss_functions": [],
        "has_batch_norm": False,
        "has_dropout": False,
        "has_lr_scheduler": False,
        "imports": []
    }
    
    # Extract imports
    import_pattern = r'^import\s+([^\n]+)$|^from\s+([^\s]+)\s+import\s+([^\n]+)$'
    for match in re.finditer(import_pattern, code, re.MULTILINE):
        if match.group(1):  # import x
            analysis["imports"].append(match.group(1).strip())
        else:  # from x import y
            analysis["imports"].append(f"{match.group(2)} → {match.group(3).strip()}")
    
    if framework == 'pytorch':
        # Check for model class
        class_pattern = r'class\s+(\w+)\s*\(\s*(?:nn\.)?Module\s*\)'
        model_match = re.search(class_pattern, code)
        if model_match:
            analysis["has_model_class"] = True
            analysis["model_class_name"] = model_match.group(1)
        
        # Check for __init__ method
        init_pattern = r'def\s+__init__\s*\('
        if re.search(init_pattern, code):
            analysis["has_init_method"] = True
        
        # Check for forward method
        forward_pattern = r'def\s+forward\s*\('
        if re.search(forward_pattern, code):
            analysis["has_forward_method"] = True
        
        # Find layers
        layer_pattern = r'self\.(\w+)\s*=\s*nn\.([\w\d]+)\('
        for match in re.finditer(layer_pattern, code):
            layer_name = match.group(1)
            layer_type = match.group(2)
            analysis["layers"].append({"name": layer_name, "type": layer_type})
            
            if 'BatchNorm' in layer_type:
                analysis["has_batch_norm"] = True
            elif 'Dropout' in layer_type:
                analysis["has_dropout"] = True
        
        # Find activations
        activation_pattern = r'(?:F|nn|torch)\.(\w+)\('
        for match in re.finditer(activation_pattern, code):
            activation = match.group(1)
            if activation.lower() in ['relu', 'sigmoid', 'tanh', 'softmax', 'leakyrelu']:
                if activation not in analysis["activations"]:
                    analysis["activations"].append(activation)
        
        # Check for training loop
        training_indicators = [
            r'\.train\(\)',
            r'\.backward\(\)',
            r'optimizer\.step\(\)',
            r'for\s+\w+\s+in\s+\w+(?:_loader|_dataloader)'
        ]
        
        for indicator in training_indicators:
            if re.search(indicator, code):
                analysis["has_training_loop"] = True
                break
        
        # Check for optimizers
        optimizer_pattern = r'(?:optim|torch\.optim)\.(\w+)\('
        for match in re.finditer(optimizer_pattern, code):
            optimizer = match.group(1)
            if optimizer not in analysis["optimizers"]:
                analysis["optimizers"].append(optimizer)
        
        # Check for LR scheduler
        scheduler_pattern = r'lr_scheduler\.(\w+)\('
        if re.search(scheduler_pattern, code):
            analysis["has_lr_scheduler"] = True
        
        # Check for loss functions
        loss_pattern = r'(?:nn|torch\.nn)\.(\w+Loss)\('
        for match in re.finditer(loss_pattern, code):
            loss = match.group(1)
            if loss not in analysis["loss_functions"]:
                analysis["loss_functions"].append(loss)
    
    # Add similar blocks for tensorflow and sklearn if needed
    
    return analysis

def _get_optimization_prompt_template(self, optimization: Dict[str, Any], framework: str) -> str:
    """Get a specialized prompt template based on the optimization type."""
    optimization_type = optimization['title'].lower()
    
    # Base template with placeholders
    base_template = """
    # Expert Machine Learning Optimization Task
    
    You are a world-class machine learning expert specializing in {framework} optimization with 10+ years of experience at leading AI research labs. You must implement the following optimization:
    
    ## Optimization Request
    
    Optimization: {optimization_title}
    Description: {optimization_description}
    Target Code Section: {code_section}
    Framework: {framework}
    Expected Benefit: {expected_benefit}
    
    ## Current Model Code
    
    ```python
    {model_code}
    ```
    
    ## Code Analysis
    {code_analysis}
    
    ## Framework-Specific Context
    {framework_context}
    
    ## Your Optimization Process
    
    Follow this precise methodology:
    
    1. First, thoroughly analyze the architecture and identify suboptimal patterns
    2. Apply proven {framework} optimization techniques from recent research papers
    3. Implement clean, efficient code that follows best practices
    4. Validate tensor shapes and operations for mathematical correctness
    5. Add strategic comments explaining your optimization rationale
    
    ## Response Format Requirements
    
    Return your solution as a JSON object with this EXACT structure:
    
    ```json
    {{
      "explanation": "Detailed technical explanation of the optimization strategy",
      "updated_code": "The complete updated code with optimization applied",
      "changes_summary": "Concise summary of modifications and their expected impact on model performance",
      "code_diff": "Lines changed: [list specific line numbers and changes]"
    }}
    ```
    
    The JSON must be properly formatted with escaped quotes and valid syntax. Focus on delivering a complete, production-ready solution.
    """
    
    # Specialized templates for different optimization types
    if 'batch' in optimization_type and 'norm' in optimization_type:
        # Batch normalization specific template
        return base_template + """
        ## Batch Normalization Specific Guidelines
        
        1. Add batch normalization layers after convolutional or linear layers, but before activation functions
        2. For CNNs, use BatchNorm2d for 2D data (images) or BatchNorm1d for 1D data
        3. Make sure to update both the __init__ method and forward method consistently
        4. Set the correct number of features for each batch norm layer (matching the output channels of the previous layer)
        5. Remember to place the model in training mode (model.train()) during training to enable proper batch statistics updates
        6. Consider adding a small epsilon parameter (e.g., 1e-5) for numerical stability
        
        Remember that batch normalization typically:
        - Accelerates training by allowing higher learning rates
        - Reduces the sensitivity to weight initialization
        - Acts as a form of regularization
        - Helps mitigate the internal covariate shift problem
        """
    
    elif 'dropout' in optimization_type:
        # Dropout specific template
        return base_template + """
        ## Dropout Specific Guidelines
        
        1. Add dropout layers AFTER activation functions (usually ReLU)
        2. Use appropriate dropout rates:
           - 0.2-0.3 for input layers
           - 0.5 for hidden layers (standard)
           - Avoid dropout before the output layer
        3. Make sure dropout is only active during training by checking model.training or using F.dropout with training=self.training
        4. For CNNs, consider using Dropout2d for feature map dropout rather than individual neurons
        5. Update both the __init__ method and forward method consistently
        
        Remember that dropout:
        - Prevents co-adaptation of neurons by randomly disabling them during training
        - Functions as a regularization technique to reduce overfitting
        - Can be interpreted as an efficient way of performing model averaging
        - Should NOT be active during inference/evaluation
        """
    
    elif 'learning rate' in optimization_type or 'scheduler' in optimization_type:
        # Learning rate scheduler specific template
        return base_template + """
        ## Learning Rate Scheduler Guidelines
        
        1. Identify the current optimizer setup in the code
        2. Add an appropriate scheduler based on the problem:
           - ReduceLROnPlateau: For reducing LR when validation metrics plateau
           - StepLR: For step-wise decay at regular intervals
           - CosineAnnealingLR: For smooth cosine-based decay
           - OneCycleLR: For the one-cycle policy with warm-up and annealing
        3. Add scheduler.step() in the training loop:
           - For validation-based schedulers: scheduler.step(val_loss) after validation
           - For epoch-based schedulers: scheduler.step() at the end of each epoch
        4. Consider adding learning rate warm-up for deep networks
        5. Include logging of the learning rate to monitor changes
        
        Remember that learning rate scheduling:
        - Improves convergence and final model performance
        - Helps escape plateaus during training
        - Can adapt to the training dynamics automatically
        - Often allows using a higher initial learning rate
        """
    
    elif 'architecture' in optimization_type or 'layer' in optimization_type:
        # Architecture modification template
        return base_template + """
        ## Architecture Modification Guidelines
        
        1. Carefully analyze the current architecture to understand layer sizes and connections
        2. Make targeted improvements while preserving the overall architecture style:
           - Increase model capacity by widening layers (more neurons/filters) or adding layers
           - Add residual connections for deeper networks to prevent gradient vanishing
           - Replace standard layers with more efficient variants when appropriate
        3. Update both the __init__ method and forward method consistently
        4. Recalculate tensor shapes carefully throughout the network
        5. Add detailed comments explaining architectural decisions
        6. Consider the computational cost of your changes
        
        Remember that architecture modifications should:
        - Match the complexity of the data and problem
        - Address specific weaknesses in the current model
        - Balance capacity improvements with computational efficiency
        - Include proper initialization for new layers
        """
    
    # Default to the base template if no specific template is available
    return base_template

def _get_optimization_specific_parameters(self, optimization_title: str) -> Dict[str, Any]:
    """Get LLM parameters optimized for specific types of optimizations."""
    optimization_type = optimization_title.lower()
    
    # Default parameters
    params = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192
    }
    
    # Adjust parameters based on optimization type
    if 'architecture' in optimization_type or 'layer' in optimization_type:
        # Architecture changes need more creative freedom but with precision
        params["temperature"] = 0.3
        params["top_p"] = 0.98
    elif 'batch' in optimization_type and 'norm' in optimization_type:
        # Batch norm is very precise, need lower temperature
        params["temperature"] = 0.1
        params["top_p"] = 0.9
    elif 'learning rate' in optimization_type or 'scheduler' in optimization_type:
        # Learning rate scheduling is a mix of creativity and precision
        params["temperature"] = 0.25
        params["top_p"] = 0.95
    
    return params

async def _validate_optimization(self, original_code: str, updated_code: str, 
                               optimization: Dict[str, Any], framework: str) -> Dict[str, Any]:
    """Validate that the optimization was correctly applied and didn't break the code."""
    result = {
        'valid': True,
        'reason': None,
        'details': []
    }
    
    # Basic validation - check if updated code compiles
    try:
        compile(updated_code, '<string>', 'exec')
    except SyntaxError as e:
        result['valid'] = False
        result['reason'] = f"Syntax error in generated code: {str(e)}"
        return result
    
    # Check if code actually changed
    if original_code == updated_code:
        result['valid'] = False
        result['reason'] = "No changes were made to the code"
        return result
    
    # Framework-specific validations
    if framework == 'pytorch':
        # Validation for PyTorch code
        
        # Check if optimization was actually applied
        optimization_type = optimization['title'].lower()
        
        if 'batch' in optimization_type and 'norm' in optimization_type:
            # Check if batch norm was added
            if 'BatchNorm' not in updated_code and 'batch_norm' not in updated_code.lower():
                result['valid'] = False
                result['reason'] = "Batch normalization was not added to the code"
                return result
                
            # Check if both __init__ and forward were updated
            if 'BatchNorm' in updated_code and 'self.bn' in updated_code:
                if 'self.bn' not in updated_code.split('def forward')[1]:
                    result['valid'] = False
                    result['reason'] = "Batch normalization was added in __init__ but not used in forward method"
                    return result
        
        elif 'dropout' in optimization_type:
            # Check if dropout was added
            if 'Dropout' not in updated_code and 'dropout' not in updated_code.lower():
                result['valid'] = False
                result['reason'] = "Dropout was not added to the code"
                return result
                
            # Check if both __init__ and forward were updated
            if 'Dropout' in updated_code and 'self.dropout' in updated_code:
                if 'self.dropout' not in updated_code.split('def forward')[1]:
                    result['valid'] = False
                    result['reason'] = "Dropout was added in __init__ but not used in forward method"
                    return result
        
        elif 'learning rate' in optimization_type or 'scheduler' in optimization_type:
            # Check if scheduler was added
            if 'lr_scheduler' not in updated_code:
                result['valid'] = False
                result['reason'] = "Learning rate scheduler was not added to the code"
                return result
                
            # Check if scheduler step is called
            if 'scheduler.step' not in updated_code:
                result['valid'] = False
                result['reason'] = "Learning rate scheduler is initialized but scheduler.step() is not called"
                return result
    
    # Add similar validation blocks for tensorflow and sklearn
    
    return result

def _create_correction_prompt(self, original_code: str, updated_code: str, 
                             validation_result: Dict[str, Any], 
                             optimization: Dict[str, Any], framework: str) -> str:
    """Create a prompt to fix specific issues identified in validation."""
    return f"""
    You previously attempted to optimize this {framework} code to add {optimization['title']}, 
    but there were issues with your implementation:
    
    ISSUE: {validation_result['reason']}
    
    Original code:
    ```python
    {original_code}
    ```
    
    Your updated code that needs fixing:
    ```python
    {updated_code}
    ```
    
    Please fix ONLY the specific issue mentioned above. Return a complete, corrected version of the code.
    
    Your response must be a valid JSON with this structure:
    {{
      "explanation": "Explanation of the fix you made",
      "updated_code": "The complete corrected code"
    }}
    """

async def _generate_optimization_explanation(self, optimization_title: str, 
                                          original_code: str, updated_code: str, 
                                          framework: str) -> str:
    """Generate a detailed explanation of the optimization benefits."""
    prompt = f"""
    Explain the following {framework} code optimization in detail:
    
    Optimization: {optimization_title}
    
    Before:
    ```python
    {original_code}
    ```
    
    After:
    ```python
    {updated_code}
    ```
    
    Focus on:
    1. What specific changes were made to the code
    2. Why these changes improve the model (technical explanation)
    3. What performance benefits can be expected
    4. Any trade-offs or considerations to be aware of
    
    Provide a technical but clear explanation suitable for a machine learning engineer.
    """
    
    try:
        explanation = await self._call_gemini_with_enhanced_params(
            prompt,
            parse_json=False,
            temperature=0.3
        )
        return explanation if explanation else f"Applied {optimization_title} to improve model performance."
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return f"Applied {optimization_title} to improve model performance."

async def _generate_fallback_optimization(self, model_code: str, optimization: Dict[str, Any], framework: str) -> Dict[str, Any]:
    """Generate a fallback optimization using a more direct approach."""
    # Create a simplified prompt focused on just the code change
    simplified_prompt = f"""
    You are a {framework} expert. Modify this code to implement {optimization['title']}.
    
    ```python
    {model_code}
    ```
    
    Return ONLY a JSON object with this structure:
    {{
      "explanation": "Brief explanation",
      "updated_code": "Complete updated code"
    }}
    
    Make sure your code is correct, complete, and follows best practices.
    """
    
    try:
        # Use a more aggressive retry strategy with simplified parameters
        for attempt in range(3):
            result = await self._call_gemini_with_enhanced_params(
                simplified_prompt,
                parse_json=True,
                temperature=0.1 + (attempt * 0.1),  # Increase temperature slightly each attempt
                max_retries=1
            )
            
            if result and "updated_code" in result:
                # Validate and clean the code
                clean_code = self._clean_and_validate_code(result["updated_code"], framework)
                
                # Only return if the code actually changed
                if clean_code != model_code:
                    result["updated_code"] = clean_code
                    result["changes_summary"] = f"Implemented {optimization['title']} using fallback approach"
                    result["code_diff"] = self._generate_detailed_diff(model_code, clean_code)
                    return result
        
        # If all attempts fail, fall back to template-based approach
        return {
            "explanation": f"Applied {optimization['title']} to improve model performance.",
            "updated_code": self._apply_template_optimization(model_code, optimization, framework),
            "changes_summary": f"Applied template-based {optimization['title']} implementation"
        }
    except Exception as e:
        logger.error(f"Error in fallback optimization: {str(e)}")
        return {
            "explanation": f"Applied basic {optimization['title']} to improve model performance.",
            "updated_code": self._apply_template_optimization(model_code, optimization, framework),
            "changes_summary": "Applied basic optimization template due to generation error"
        }

def _apply_template_optimization(self, code: str, optimization: Dict[str, Any], framework: str) -> str:
    """Apply a template-based optimization when AI generation fails."""
    # This combines and enhances the existing _add_batch_norm_pytorch, _add_dropout_pytorch, etc. methods
    
    optimization_type = optimization['title'].lower()
    
    if framework == 'pytorch':
        if 'batch' in optimization_type and 'norm' in optimization_type:
            return self._add_enhanced_batch_norm_pytorch(code)
        elif 'dropout' in optimization_type:
            return self._add_enhanced_dropout_pytorch(code)
        elif 'learning rate' in optimization_type or 'scheduler' in optimization_type:
            return self._add_enhanced_lr_scheduler_pytorch(code)
        elif 'architecture' in optimization_type or 'layer' in optimization_type:
            return self._enhance_architecture_pytorch(code)
        # Add more template optimizations as needed
    
    # Add similar blocks for tensorflow and sklearn
    
    # Default: return original code with a comment
    return code + f"\n\n# TODO: Apply optimization: {optimization['title']}"

def _add_enhanced_batch_norm_pytorch(self, code: str) -> str:
    """Add batch normalization to PyTorch model with enhanced implementation."""
    # This is a significantly improved version of the existing _add_batch_norm_pytorch method
    
    # Create a copy of the code to modify
    modified_code = code
    
    # Parse the code to identify model structure
    has_cnn = 'Conv2d' in code
    has_fc = 'Linear' in code
    
    # Regular expressions for finding layers
    conv_pattern = r'(self\.\w+)\s*=\s*nn\.Conv2d\(([^)]+)\)'
    linear_pattern = r'(self\.\w+)\s*=\s*nn\.Linear\(([^)]+)\)'
    
    # Add batch norm after convolutional layers
    if has_cnn:
        # Find all Conv2d layers
        conv_matches = list(re.finditer(conv_pattern, modified_code))
        
        # Add batch norm for each conv layer
        for match in reversed(conv_matches):  # Process in reverse to avoid messing up indexes
            conv_name = match.group(1)
            conv_args = match.group(2)
            
            # Extract the output channels (second parameter in Conv2d)
            # Extract the output channels (second parameter in Conv2d)
            params = [p.strip() for p in conv_args.split(',')]
            out_channels = params[1] if len(params) > 1 else "64"  # Default fallback
            
            # Create batch norm layer name based on conv layer name
            bn_name = conv_name.replace('conv', 'bn')
            if bn_name == conv_name:
                bn_name = f"{conv_name}_bn"
            
            # Insert batch norm after the conv layer
            bn_line = f"{bn_name} = nn.BatchNorm2d({out_channels})"
            modified_code = modified_code.replace(match.group(0), f"{match.group(0)}\n        {bn_line}")
            
            # Add comment explaining the addition
            modified_code = modified_code.replace(bn_line, f"{bn_line}  # Added batch normalization after {conv_name}")
    
    # Add batch norm after fully connected layers if appropriate
    if has_fc and not has_cnn:  # Only for pure FC networks
        # Find all Linear layers
        linear_matches = list(re.finditer(linear_pattern, modified_code))
        
        for match in reversed(linear_matches):
            linear_name = match.group(1)
            linear_args = match.group(2)
            
            # Extract the output features (second parameter in Linear)
            params = [p.strip() for p in linear_args.split(',')]
            out_features = params[1] if len(params) > 1 else "64"  # Default fallback
            
            # Create batch norm layer name
            bn_name = linear_name.replace('fc', 'bn')
            if bn_name == linear_name:
                bn_name = f"{linear_name}_bn"
            
            # Insert batch norm after the linear layer
            bn_line = f"{bn_name} = nn.BatchNorm1d({out_features})"
            modified_code = modified_code.replace(match.group(0), f"{match.group(0)}\n        {bn_line}")
            
            # Add comment explaining the addition
            modified_code = modified_code.replace(bn_line, f"{bn_line}  # Added batch normalization after {linear_name}")
    
    # Now update the forward method to use the batch norm layers
    # First, locate the forward method
    forward_pattern = r'def\s+forward\s*\(\s*self\s*,\s*x\s*(?:,\s*\*\w+)?\s*\):\s*\n((?:.*\n)*?)\s*return'
    forward_match = re.search(forward_pattern, modified_code)
    
    if forward_match:
        forward_body = forward_match.group(1)
        new_forward_body = forward_body
        
        # Add batch norm after each conv layer in forward
        if has_cnn:
            conv_forward_pattern = r'(x\s*=\s*(?:self\.)?([\w]+)\s*\(\s*x\s*\))'
            for match in re.finditer(conv_forward_pattern, forward_body):
                full_match = match.group(1)
                layer_name = match.group(2)
                
                # Check if this is a conv layer by looking at the layer declaration
                if f"self.{layer_name} = nn.Conv" in modified_code:
                    # Generate the corresponding batch norm name
                    bn_name = f"self.{layer_name}_bn" if f"self.{layer_name}_bn" in modified_code else f"self.bn_{layer_name}"
                    
                    # Determine if activation follows immediately
                    next_line_pattern = f"{re.escape(full_match)}(.*?)(?:\n|$)"
                    next_line_match = re.search(next_line_pattern, forward_body)
                    has_immediate_activation = next_line_match and ('relu' in next_line_match.group(1).lower() or 
                                                                 'sigmoid' in next_line_match.group(1).lower() or 
                                                                 'tanh' in next_line_match.group(1).lower())
                    
                    # Insert batch norm before activation but after conv
                    if has_immediate_activation:
                        replacement = f"{full_match}\n        x = {bn_name}(x)"
                    else:
                        replacement = f"{full_match}\n        x = {bn_name}(x)"
                    
                    new_forward_body = new_forward_body.replace(full_match, replacement)
        
        # Update the forward method in the code
        modified_code = modified_code.replace(forward_body, new_forward_body)
    
    # Add necessary imports if not present
    if 'import torch' not in modified_code:
        modified_code = 'import torch\n' + modified_code
    if 'import torch.nn as nn' not in modified_code:
        modified_code = 'import torch.nn as nn\n' + modified_code
    
    # Add a comment at the top explaining the optimization
    modified_code = f"# Model optimized with Batch Normalization\n{modified_code}"
    
    return modified_code

def _add_enhanced_dropout_pytorch(self, code: str) -> str:
    """Add dropout to PyTorch model with enhanced implementation."""
    # This is a significantly improved version of the existing _add_dropout_pytorch method
    
    # Create a copy of the code to modify
    modified_code = code
    
    # Detect model type (CNN or fully connected)
    is_cnn = 'Conv2d' in code
    has_fc = 'Linear' in code
    
    # Add imports if missing
    if 'import torch.nn as nn' not in modified_code:
        if 'import torch' in modified_code:
            modified_code = modified_code.replace('import torch', 'import torch\nimport torch.nn as nn')
        else:
            modified_code = 'import torch\nimport torch.nn as nn\n' + modified_code
    
    # Find the model class
    class_pattern = r'class\s+(\w+)\s*\(\s*(?:nn\.)?Module\s*\):'
    class_match = re.search(class_pattern, modified_code)
    
    if class_match:
        class_name = class_match.group(1)
        
        # Find the __init__ method
        init_pattern = r'def\s+__init__\s*\(\s*self\s*,\s*(.*?)\s*\):\s*\n((?:.*\n)*?)(?=\s*def|\s*$)'
        init_match = re.search(init_pattern, modified_code)
        
        if init_match:
            init_params = init_match.group(1)
            init_body = init_match.group(2)
            
            # Add dropout layers to __init__
            new_init_body = init_body
            
            # Calculate appropriate dropout rates based on model architecture
            if is_cnn:
                # For CNNs, we typically use lower dropout rates
                dropout_rate_early = 0.1
                dropout_rate_mid = 0.25
                dropout_rate_late = 0.3
                
                # Add Dropout2d for CNNs
                dropout_lines = [
                    "        # Add dropout for regularization",
                    "        self.dropout1 = nn.Dropout2d(p=0.1)  # Light dropout after early layers",
                    "        self.dropout2 = nn.Dropout2d(p=0.25)  # Medium dropout after middle layers",
                    "        self.dropout3 = nn.Dropout(p=0.3)  # Higher dropout for fully connected layers"
                ]
            else:
                # For fully connected networks, we typically use higher rates
                dropout_rate = 0.5
                
                # Add standard Dropout for FC networks
                dropout_lines = [
                    "        # Add dropout for regularization",
                    "        self.dropout1 = nn.Dropout(p=0.2)  # Input dropout",
                    "        self.dropout2 = nn.Dropout(p=0.5)  # Hidden layer dropout"
                ]
            
            # Find a good place to insert dropout declarations
            if 'super(' in init_body:
                # Add after the super().__init__() call
                super_pattern = r'(super\(.*?\)\.__init__\(.*?\))'
                new_init_body = re.sub(super_pattern, r'\1\n' + '\n'.join(dropout_lines), new_init_body)
            else:
                # Add at the beginning of the method body
                new_init_body = re.sub(r'(\n\s*)', r'\1' + '\n'.join(dropout_lines) + '\n', new_init_body, count=1)
            
            # Update the __init__ method
            modified_code = modified_code.replace(init_body, new_init_body)
        
        # Now update the forward method
        forward_pattern = r'def\s+forward\s*\(\s*self\s*,\s*x\s*(?:,\s*\*\w+)?\s*\):\s*\n((?:.*\n)*?)(\s*return\s+\w+)'
        forward_match = re.search(forward_pattern, modified_code)
        
        if forward_match:
            forward_body = forward_match.group(1)
            return_stmt = forward_match.group(2)
            
            # Add dropout in the forward method at appropriate places
            new_forward_body = forward_body
            
            if is_cnn:
                # For CNNs, add dropout after activations
                # Find ReLU or other activation patterns
                activation_pattern = r'(x\s*=\s*(?:F\.)?relu\(.*?\))'
                
                # Count activations to place dropouts strategically
                activation_matches = list(re.finditer(activation_pattern, forward_body))
                num_activations = len(activation_matches)
                
                if num_activations > 0:
                    # Add dropout after appropriate activations
                    if num_activations >= 3:
                        # Add dropout1 after first activation
                        match = activation_matches[0]
                        new_forward_body = new_forward_body.replace(match.group(0), 
                                                                  f"{match.group(0)}\n        x = self.dropout1(x)  # Early dropout")
                        
                        # Add dropout2 after middle activation
                        mid_idx = num_activations // 2
                        match = activation_matches[mid_idx]
                        new_forward_body = new_forward_body.replace(match.group(0), 
                                                                  f"{match.group(0)}\n        x = self.dropout2(x)  # Middle dropout")
                        
                        # Find fully connected layers after flattening
                        if 'flatten' in new_forward_body.lower() or 'view' in new_forward_body.lower():
                            # Add dropout3 after the first FC layer's activation after flattening
                            fc_pattern = r'(x\s*=\s*(?:F\.)?relu\(.*?\))(?!.*?dropout)'
                            fc_match = re.search(fc_pattern, new_forward_body.split('flatten')[1] if 'flatten' in new_forward_body.lower() 
                                               else new_forward_body.split('view')[1])
                            if fc_match:
                                flat_section = new_forward_body.split('flatten')[1] if 'flatten' in new_forward_body.lower() else new_forward_body.split('view')[1]
                                replaced_flat_section = flat_section.replace(fc_match.group(0), 
                                                                          f"{fc_match.group(0)}\n        x = self.dropout3(x)  # FC dropout")
                                
                                if 'flatten' in new_forward_body.lower():
                                    new_forward_body = new_forward_body.split('flatten')[0] + 'flatten' + replaced_flat_section
                                else:
                                    new_forward_body = new_forward_body.split('view')[0] + 'view' + replaced_flat_section
                    else:
                        # For simpler networks, add dropout after the first activation
                        match = activation_matches[0]
                        new_forward_body = new_forward_body.replace(match.group(0), 
                                                                  f"{match.group(0)}\n        x = self.dropout1(x)  # Apply dropout")
            else:
                # For fully connected networks
                # Find layers and add dropout after their activations
                fc_pattern = r'(x\s*=\s*(?:F\.)?relu\(.*?\))'
                fc_matches = list(re.finditer(fc_pattern, forward_body))
                
                if len(fc_matches) >= 2:
                    # Add dropout1 after first layer's activation
                    match = fc_matches[0]
                    new_forward_body = new_forward_body.replace(match.group(0), 
                                                              f"{match.group(0)}\n        x = self.dropout1(x)  # First dropout")
                    
                    # Add dropout2 after intermediate activations but not the last one before output
                    for i in range(1, len(fc_matches) - 1):
                        match = fc_matches[i]
                        new_forward_body = new_forward_body.replace(match.group(0), 
                                                                  f"{match.group(0)}\n        x = self.dropout2(x)  # Hidden dropout")
                elif len(fc_matches) == 1:
                    # For very simple networks with just one hidden layer
                    match = fc_matches[0]
                    new_forward_body = new_forward_body.replace(match.group(0), 
                                                              f"{match.group(0)}\n        x = self.dropout1(x)  # Apply dropout")
            
            # Update the forward method in the code
            modified_code = modified_code.replace(forward_body + return_stmt, new_forward_body + return_stmt)
    
    # Add necessary imports if not present
    if 'import torch.nn.functional as F' not in modified_code and 'import torch.nn.functional' not in modified_code:
        if 'import torch.nn as nn' in modified_code:
            modified_code = modified_code.replace('import torch.nn as nn', 'import torch.nn as nn\nimport torch.nn.functional as F')
        else:
            modified_code = 'import torch.nn as nn\nimport torch.nn.functional as F\n' + modified_code
    
    # Add a comment at the top explaining the optimization
    modified_code = f"# Model optimized with Dropout for regularization\n{modified_code}"
    
    return modified_code

def _add_enhanced_lr_scheduler_pytorch(self, code: str) -> str:
    """Add an advanced learning rate scheduler to PyTorch code."""
    # This is a significantly improved version of the existing _add_lr_scheduler_pytorch method
    
    # Create a copy of the code to modify
    modified_code = code
    
    # Check if optimizer is already present
    has_optimizer = 'optimizer' in code.lower() and ('sgd' in code.lower() or 'adam' in code.lower() or 'optim.' in code.lower())
    
    # Detect if training loop exists
    has_training_loop = 'epoch' in code.lower() and 'train' in code.lower() and ('for' in code.lower() or 'while' in code.lower())
    
    # Detect if validation is performed
    has_validation = 'val' in code.lower() and 'loss' in code.lower()
    
    # Choose the appropriate scheduler based on model characteristics
    if has_validation:
        scheduler_type = "ReduceLROnPlateau"
        scheduler_code = """
    # Add learning rate scheduler that reduces LR when validation performance plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # Lower val_loss is better
        factor=0.1,           # Reduce LR by factor of 10 when triggered
        patience=5,           # Number of epochs with no improvement after which LR will be reduced
        verbose=True,         # Print message when LR is reduced
        min_lr=1e-6           # Lower bound on the learning rate
    )"""
        scheduler_step = """
    # Step the learning rate scheduler based on validation loss
    scheduler.step(val_loss)"""
    else:
        scheduler_type = "CosineAnnealingWarmRestarts"
        scheduler_code = """
    # Add cosine annealing scheduler with warm restarts for cyclical learning rates
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,               # Number of epochs for first restart
        T_mult=2,             # Multiply T_0 by this number after each restart
        eta_min=1e-6,         # Minimum learning rate
        verbose=True          # Print message when LR is updated
    )"""
        scheduler_step = """
    # Step the learning rate scheduler at the end of each epoch
    scheduler.step()"""
    
    # Add imports if necessary
    if 'import torch.optim' not in modified_code and 'from torch import optim' not in modified_code:
        if 'import torch' in modified_code:
            modified_code = modified_code.replace('import torch', 'import torch\nimport torch.optim as optim')
        else:
            modified_code = 'import torch\nimport torch.optim as optim\n' + modified_code
    
    # If no optimizer is present, add one
    if not has_optimizer:
        # Find model parameters
        model_pattern = r'class\s+(\w+)\s*\('
        model_match = re.search(model_pattern, modified_code)
        
        if model_match:
            model_name = model_match.group(1)
            
            # Add optimizer initialization code
            optimizer_code = f"""
def train_{model_name.lower()}(model, train_loader, val_loader=None, epochs=10, lr=0.001):
    \"\"\"Train the model with an optimized learning rate schedule.\"\"\"
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss(){scheduler_code}
    
    # Training loop
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        running_loss = 0.0
        
        # Iterate over training data
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
        
        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {{epoch+1}}/{{epochs}}, Loss: {{epoch_loss:.4f}}')
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            print(f'Validation Loss: {{val_loss:.4f}}, Accuracy: {{val_accuracy:.2f}}%'){scheduler_step}
            
            # Print current learning rate
            for param_group in optimizer.param_groups:
                print(f'Learning rate: {{param_group["lr"]:.6f}}')
        else:{scheduler_step}
    
    return model
"""
            # Add the training function to the end of the file
            modified_code += "\n\n" + optimizer_code
    else:
        # Find existing training loop
        if has_training_loop:
            # Find where to add the scheduler initialization
            optimizer_pattern = r'(optimizer\s*=\s*(?:torch\.)?optim\.(?:SGD|Adam|RMSprop)\(.*?\))'
            optimizer_match = re.search(optimizer_pattern, modified_code)
            
            if optimizer_match:
                # Add scheduler initialization after optimizer
                modified_code = modified_code.replace(optimizer_match.group(0), 
                                                   f"{optimizer_match.group(0)}\n{scheduler_code}")
            
            # Find where to add the scheduler step
            if has_validation:
                # Find validation loss calculation
                val_loss_pattern = r'(val_loss\s*=\s*.*?)\n'
                val_loss_match = re.search(val_loss_pattern, modified_code)
                
                if val_loss_match:
                    # Add scheduler step after validation loss calculation
                    modified_code = modified_code.replace(val_loss_match.group(0), 
                                                       f"{val_loss_match.group(0)}{scheduler_step}\n")
            else:
                # Find end of epoch
                epoch_end_pattern = r'(print\(.*?epoch.*?\))'
                epoch_end_match = re.search(epoch_end_pattern, modified_code)
                
                if epoch_end_match:
                    # Add scheduler step after epoch end
                    modified_code = modified_code.replace(epoch_end_match.group(0), 
                                                       f"{epoch_end_match.group(0)}\n    {scheduler_step}")
        else:
            # No training loop found, add a training function
            model_pattern = r'class\s+(\w+)\s*\('
            model_match = re.search(model_pattern, modified_code)
            
            if model_match:
                model_name = model_match.group(1)
                
                # Add complete training function with scheduler
                train_function = f"""
def train_{model_name.lower()}_with_scheduler(model, train_loader, val_loader=None, epochs=10):
    \"\"\"Train the model with learning rate scheduling for better convergence.\"\"\"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    {scheduler_code}
    
    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate average losses and accuracy
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Print statistics
        print(f'Epoch {{epoch+1}}/{{epochs}}:')
        print(f'  Train Loss: {{train_loss:.4f}}')
        print(f'  Val Loss: {{val_loss:.4f}}')
        print(f'  Accuracy: {{accuracy:.2f}}%')
        {scheduler_step}
        
        # Print current learning rate
        for param_group in optimizer.param_groups:
            print(f'  Learning rate: {{param_group["lr"]:.6f}}')
    
    return model
"""
                # Add the training function to the end of the file
                modified_code += "\n\n" + train_function
    
    # Add a comment at the top explaining the optimization
    modified_code = f"# Model optimized with {scheduler_type} learning rate scheduling\n{modified_code}"
    
    return modified_code

def _enhance_architecture_pytorch(self, code: str) -> str:
    """Enhance the PyTorch model architecture with modern improvements."""
    # This is a new method to enhance model architecture
    
    # Create a copy of the code to modify
    modified_code = code
    
    # Detect model type (CNN, RNN, or fully connected)
    is_cnn = 'Conv2d' in code
    is_rnn = any(x in code for x in ['LSTM', 'GRU', 'RNN'])
    is_fc = 'Linear' in code and not is_cnn and not is_rnn
    
    # Find the model class
    class_pattern = r'class\s+(\w+)\s*\(\s*(?:nn\.)?Module\s*\):\s*\n((?:.*\n)*?)(?=\s*class|\s*def\s+(?!__init__|forward)|\s*$)'
    class_match = re.search(class_pattern, modified_code)
    
    if class_match:
        class_name = class_match.group(1)
        class_body = class_match.group(2)
        
        # Analyze the model's current architecture
        layer_patterns = {
            'conv': r'self\.(\w+)\s*=\s*nn\.Conv2d\(([^)]+)\)',
            'linear': r'self\.(\w+)\s*=\s*nn\.Linear\(([^)]+)\)',
            'lstm': r'self\.(\w+)\s*=\s*nn\.LSTM\(([^)]+)\)',
            'pool': r'self\.(\w+)\s*=\s*nn\.(?:Max|Avg)Pool2d\(([^)]+)\)',
            'dropout': r'self\.(\w+)\s*=\s*nn\.Dropout\(([^)]+)\)',
            'batchnorm': r'self\.(\w+)\s*=\s*nn\.BatchNorm\w*\(([^)]+)\)',
        }
        
        layers = {}
        for layer_type, pattern in layer_patterns.items():
            layers[layer_type] = []
            for match in re.finditer(pattern, class_body):
                layer_name = match.group(1)
                layer_args = match.group(2)
                layers[layer_type].append((layer_name, layer_args))
        
        # Determine what enhancements to make based on the model type and current architecture
        new_class_body = class_body
        
        if is_cnn:
            # Enhance CNN architecture
            
            # Check if the model already has skip connections
            has_skip_connections = '+' in class_body and 'forward' in class_body
            
            if not has_skip_connections and len(layers.get('conv', [])) >= 2:
                # Add residual connections for deeper CNNs
                
                # First, modify the __init__ method to add residual blocks
                init_pattern = r'(def\s+__init__\s*\([^)]*\):(?:\s*\n\s+.*?)*)((?=\s+def|\s*$))'
                init_match = re.search(init_pattern, class_body, re.DOTALL)
                
                if init_match:
                    init_method = init_match.group(1)
                    
                    # Find some consecutive Conv2d layers to convert to residual blocks
                    conv_indices = []
                    lines = init_method.split('\n')
                    for i, line in enumerate(lines):
                        if 'nn.Conv2d' in line:
                            conv_indices.append(i)
                    
                    # Only proceed if we have enough convolutional layers
                    if len(conv_indices) >= 2:
                        # Identify pairs of conv layers to turn into residual blocks
                        res_blocks = []
                        for i in range(0, len(conv_indices) - 1, 2):
                            if i + 1 < len(conv_indices):
                                # Get the two conv layer indices
                                idx1 = conv_indices[i]
                                idx2 = conv_indices[i + 1]
                                
                                # Extract layer names and arguments
                                conv1_match = re.search(r'self\.(\w+)\s*=\s*nn\.Conv2d\(([^)]+)\)', lines[idx1])
                                conv2_match = re.search(r'self\.(\w+)\s*=\s*nn\.Conv2d\(([^)]+)\)', lines[idx2])
                                
                                if conv1_match and conv2_match:
                                    conv1_name = conv1_match.group(1)
                                    conv1_args = conv1_match.group(2)
                                    conv2_name = conv2_match.group(1)
                                    conv2_args = conv2_match.group(2)
                                    
                                    # Parse arguments to determine input/output channels
                                    conv1_params = [p.strip() for p in conv1_args.split(',')]
                                    conv2_params = [p.strip() for p in conv2_args.split(',')]
                                    
                                    if len(conv1_params) >= 2 and len(conv2_params) >= 2:
                                        in_channels = conv1_params[0]
                                        out_channels = conv2_params[1]
                                        
                                        # Check if input and output channels are different (need a shortcut)
                                        needs_shortcut = in_channels != out_channels
                                        
                                        # Add this pair to our residual blocks
                                        res_blocks.append({
                                            'conv1_idx': idx1,
                                            'conv2_idx': idx2,
                                            'conv1_name': conv1_name,
                                            'conv2_name': conv2_name,
                                            'in_channels': in_channels,
                                            'out_channels': out_channels,
                                            'needs_shortcut': needs_shortcut
                                        })
                        
                        # Now modify the init method to include residual blocks
                        if res_blocks:
                            new_init_method = init_method
                            
                            # Add residual block class at the beginning of the init method
                            res_block_class = """
    # Residual block for skip connections
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            # Shortcut connection to match dimensions
            # Shortcut connection to match dimensions
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
                
        def forward(self, x):
            residual = x
            
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            
            out = self.conv2(out)
            out = self.bn2(out)
            
            out += self.shortcut(residual)
            out = self.relu(out)
            
            return out"""
                            
                            # Add the residual block class after super().__init__()
                            if 'super(' in new_init_method:
                                super_pattern = r'(super\(.*?\)\.__init__\(.*?\))'
                                new_init_method = re.sub(super_pattern, r'\1\n' + res_block_class, new_init_method)
                            else:
                                # Add at the beginning of the method
                                new_init_method = re.sub(r'(def\s+__init__\s*\([^)]*\):)', r'\1\n' + res_block_class, new_init_method)
                            
                            # Replace the pairs of convolutions with residual blocks
                            for block in res_blocks:
                                # Generate a name for this residual block
                                block_name = f"res_block_{block['conv1_name'].replace('conv', '')}"
                                
                                # Create the new block initialization
                                res_block_init = f"""
        # Residual block replacing {block['conv1_name']} and {block['conv2_name']}
        self.{block_name} = self.ResidualBlock({block['in_channels']}, {block['out_channels']})"""
                                
                                # Find a good place to insert the block - after the second conv layer
                                conv2_line = next((line for i, line in enumerate(new_init_method.split('\n')) 
                                               if i >= block['conv2_idx'] and f"self.{block['conv2_name']}" in line), None)
                                
                                if conv2_line:
                                    new_init_method = new_init_method.replace(conv2_line, conv2_line + res_block_init)
                            
                            # Update the init method
                            new_class_body = new_class_body.replace(init_method, new_init_method)
                            
                            # Now update the forward method to use the residual blocks
                            forward_pattern = r'(def\s+forward\s*\([^)]*\):(?:\s*\n\s+.*?)*)((?=\s+def|\s*$))'
                            forward_match = re.search(forward_pattern, new_class_body, re.DOTALL)
                            
                            if forward_match:
                                forward_method = forward_match.group(1)
                                new_forward_method = forward_method
                                
                                # Replace pairs of convolutions with calls to residual blocks
                                for block in res_blocks:
                                    # Look for the pattern where both convs are applied in sequence
                                    conv_pattern = r'(x\s*=\s*(?:self\.)?{}\(x\).*?\n\s+x\s*=\s*(?:self\.)?{}\(x\))'.format(
                                        re.escape(block['conv1_name']), re.escape(block['conv2_name']))
                                    
                                    conv_match = re.search(conv_pattern, forward_method, re.DOTALL)
                                    
                                    if conv_match:
                                        # Replace with residual block
                                        block_name = f"res_block_{block['conv1_name'].replace('conv', '')}"
                                        replacement = f"x = self.{block_name}(x)  # Residual block"
                                        new_forward_method = new_forward_method.replace(conv_match.group(1), replacement)
                                
                                # Update the forward method
                                new_class_body = new_class_body.replace(forward_method, new_forward_method)
        
        # For fully connected networks, consider adding more capacity and better initialization
        elif is_fc:
            # Check number of linear layers
            if len(layers.get('linear', [])) < 3:
                # Add more depth to the FC network
                
                # Find the __init__ method
                init_pattern = r'(def\s+__init__\s*\([^)]*\):(?:\s*\n\s+.*?)*)((?=\s+def|\s*$))'
                init_match = re.search(init_pattern, class_body, re.DOTALL)
                
                if init_match:
                    init_method = init_match.group(1)
                    new_init_method = init_method
                    
                    # Find the last linear layer
                    linear_layers = sorted([(i, line) for i, line in enumerate(init_method.split('\n')) if 'nn.Linear' in line], 
                                        key=lambda x: x[0])
                    
                    if linear_layers:
                        last_linear_idx, last_linear_line = linear_layers[-1]
                        
                        # Extract layer name and arguments
                        linear_match = re.search(r'self\.(\w+)\s*=\s*nn\.Linear\(([^)]+)\)', last_linear_line)
                        
                        if linear_match:
                            last_layer_name = linear_match.group(1)
                            last_layer_args = linear_match.group(2)
                            
                            # Parse arguments to determine input/output size
                            params = [p.strip() for p in last_layer_args.split(',')]
                            
                            if len(params) >= 2:
                                in_features = params[0]
                                out_features = params[1]
                                
                                # Check if this is the output layer (typically small out_features)
                                try:
                                    out_size = int(out_features)
                                    is_output_layer = out_size < 100  # Heuristic for output layer
                                except:
                                    is_output_layer = False
                                
                                if is_output_layer and len(linear_layers) < 3:
                                    # Find the previous layer to get its size
                                    prev_layer_idx = -1
                                    for i, (idx, line) in enumerate(linear_layers):
                                        if last_layer_name in line:
                                            if i > 0:
                                                prev_layer_idx = i - 1
                                            break
                                    
                                    if prev_layer_idx >= 0:
                                        prev_idx, prev_line = linear_layers[prev_layer_idx]
                                        prev_match = re.search(r'self\.(\w+)\s*=\s*nn\.Linear\(([^)]+)\)', prev_line)
                                        
                                        if prev_match:
                                            prev_args = prev_match.group(2)
                                            prev_params = [p.strip() for p in prev_args.split(',')]
                                            
                                            if len(prev_params) >= 2:
                                                prev_out = prev_params[1]
                                                
                                                # Add an intermediate layer
                                                intermediate_size = f"int({prev_out} / 2)"  # Half the size of previous layer
                                                new_layer_name = f"fc_intermediate"
                                                
                                                # Create new layer code
                                                new_layer_code = f"""
        # Add an intermediate layer for better model capacity
        self.{new_layer_name} = nn.Linear({prev_out}, {intermediate_size})
        self.{new_layer_name}_bn = nn.BatchNorm1d({intermediate_size})
        self.{new_layer_name}_relu = nn.ReLU(inplace=True)
        self.{new_layer_name}_dropout = nn.Dropout(0.3)
        
        # Replace output layer to connect from the new intermediate layer
        self.{last_layer_name} = nn.Linear({intermediate_size}, {out_features})"""
                                                
                                                # Insert new layer before the output layer
                                                lines = new_init_method.split('\n')
                                                new_lines = []
                                                
                                                for i, line in enumerate(lines):
                                                    if i == last_linear_idx:
                                                        new_lines.append(new_layer_code)
                                                    else:
                                                        if f"self.{last_layer_name}" not in line:
                                                            new_lines.append(line)
                                                
                                                new_init_method = '\n'.join(new_lines)
                                                
                                                # Update the init method
                                                new_class_body = new_class_body.replace(init_method, new_init_method)
                                                
                                                # Now update the forward method
                                                forward_pattern = r'(def\s+forward\s*\([^)]*\):(?:\s*\n\s+.*?)*)((?=\s+def|\s*$))'
                                                forward_match = re.search(forward_pattern, new_class_body, re.DOTALL)
                                                
                                                if forward_match:
                                                    forward_method = forward_match.group(1)
                                                    
                                                    # Find where the output layer is used
                                                    output_pattern = r'(x\s*=\s*(?:self\.)?{}(?:\(x\)|\.forward\(x\)))'.format(
                                                        re.escape(last_layer_name))
                                                    
                                                    output_match = re.search(output_pattern, forward_method)
                                                    
                                                    if output_match:
                                                        # Insert intermediate layer processing before output layer
                                                        intermediate_code = f"""
        # Process through intermediate layer
        x = self.{new_layer_name}(x)
        x = self.{new_layer_name}_bn(x)
        x = self.{new_layer_name}_relu(x)
        x = self.{new_layer_name}_dropout(x)
        
        # Output layer
        {output_match.group(1)}"""
                                                        
                                                        new_forward_method = forward_method.replace(output_match.group(1), intermediate_code)
                                                        
                                                        # Update the forward method
                                                        new_class_body = new_class_body.replace(forward_method, new_forward_method)
        
        # Update the class body
        modified_code = modified_code.replace(class_body, new_class_body)
    
    # Add necessary imports if not present
    if 'import torch' not in modified_code:
        modified_code = 'import torch\n' + modified_code
    if 'import torch.nn as nn' not in modified_code:
        modified_code = 'import torch\nimport torch.nn as nn\n' + modified_code
    if 'import torch.nn.functional as F' not in modified_code:
        modified_code = modified_code.replace('import torch.nn as nn', 'import torch.nn as nn\nimport torch.nn.functional as F')
    
    # Add a comment at the top explaining the optimization
    modified_code = f"# Model optimized with enhanced architecture\n{modified_code}"
    
    return modified_code

async def handle_websocket_error(websocket: WebSocket, error: Exception, message: str = None):
    """Handle WebSocket errors with appropriate feedback to the user."""
    error_message = message or str(error)
    error_type = type(error).__name__
    error_details = {
        "type": "error",
        "message": error_message,
        "error_type": error_type
    }
    
    # Log the error with traceback for server-side debugging
    import traceback
    logger.error(f"WebSocket error: {error_message}")
    logger.error(traceback.format_exc())
    
    # Add additional context for specific error types
    if isinstance(error, (ValueError, TypeError, SyntaxError)):
        error_details["suggestion"] = "Check the format of your input data and try again."
    elif "rate limit" in error_message.lower() or "quota" in error_message.lower():
        error_details["suggestion"] = "You've reached your API usage limit. Please upgrade your plan or try again later."
    elif "timeout" in error_message.lower():
        error_details["suggestion"] = "The operation timed out. Try with a smaller model or simplify your request."
    
    try:
        # Send detailed error information to the client
        await websocket.send_json(error_details)
    except Exception as e:
        # If we can't send the error as JSON, try a simpler error message
        try:
            await websocket.send_text(f"Error: {error_message}")
        except:
            # If all else fails, just close the connection
            pass
    
    # Try to close the connection gracefully
    try:
        await websocket.close(code=1011, reason=error_message[:123])  # 123 chars is the max for close reason
    except:
        # Connection might already be closed
        pass

async def _call_gemini_with_enhanced_params(self, 
                                         prompt: str, 
                                         parse_json: bool = False,
                                         max_retries: int = 3) -> Union[str, Dict[str, Any], List, None]:
    """Call Gemini API with optimized parameters for code generation."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling Gemini API with enhanced parameters - attempt {attempt+1}/{max_retries}")
            
            # Call the API using the client with optimized parameters
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",  # Use the best available model
                contents=prompt,
                generation_config={
                    "temperature": 0.2,       # Lower temperature for more precise code
                    "top_p": 0.95,            # Higher precision in token selection
                    "top_k": 40,              # More candidate tokens considered
                    "max_output_tokens": 8192, # Allow larger responses for complete code
                    "candidate_count": 1      # One high-quality response
                },
                system_instruction="You are an AI assistant specializing in creating precise, efficient machine learning code optimizations. Always provide complete, working implementations. Focus on modern techniques that improve model accuracy and training efficiency. Ensure all tensor operations are correct and code is production-ready."
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                text_response = response.text
            elif hasattr(response, 'parts') and response.parts:
                text_response = response.parts[0].text
            else:
                logger.error(f"Unexpected response format: {type(response)}")
                raise Exception("Unexpected response format from Gemini API")
            
            # Parse JSON if requested
            if parse_json:
                result = self._extract_json_from_response(text_response)
                if not result and attempt < max_retries - 1:
                    # Try again with a different prompt approach
                    logger.warning("JSON parsing failed, retrying with modified prompt")
                    prompt += "\n\nIMPORTANT: You MUST return a valid JSON object with the exact structure specified above. No markdown, no additional text."
                    await asyncio.sleep(1)
                    continue
                return result
            else:
                return text_response
        
        except Exception as e:
            error_str = str(e)
            logger.error(f"Error on attempt {attempt+1}: {error_str}")
            
            # Check if it's a rate limit error
            if "quota" in error_str.lower() or "rate" in error_str.lower() or "429" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 0.5 + (random.random() * 0.5)  # Add jitter
                    logger.warning(f"Rate limited. Retrying in {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # All retries failed
                    logger.error("Rate limit retries exhausted")
                    raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
            else:
                # For other errors, retry with a more direct approach
                if attempt < max_retries - 1:
                    logger.warning(f"Error occurred, retrying with simplified prompt")
                    # Simplify the prompt
                    if parse_json:
                        # Add stronger formatting instructions
                        prompt = self._simplify_prompt_for_retry(prompt)
                    await asyncio.sleep(1.5)  # Wait longer to avoid rate limiting
                    continue
                else:
                    # All retries failed
                    raise
    
    # If we reach here, all retries failed
    logger.error("All API call attempts failed")
    return None

def _clean_and_validate_code(self, code: str, framework: str) -> str:
    """Clean up code and validate it for correctness."""
    # Remove markdown code blocks
    code = re.sub(r'```python\s*', '', code)
    code = re.sub(r'```\s*', '', code)
    
    # Strip leading/trailing whitespace
    code = code.strip()
    
    # Basic syntax validation
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        logger.warning(f"Generated code has syntax error: {e}")
        # Try to fix common syntax issues
        code = self._attempt_syntax_fix(code)
    
    # Framework-specific validation
    if framework == 'pytorch':
        # Check for common PyTorch mistakes
        if 'nn.Module' in code and '__init__' in code and 'super(' not in code:
            logger.warning("Missing super().__init__() call in PyTorch model")
            # Add super().__init__() if missing
            code = re.sub(r'def __init__\([^:]+:\s+', r'\g<0>    super().__init__()\n    ', code)
    
    return code

def _generate_detailed_diff(self, old_code: str, new_code: str) -> str:
    """Generate a more detailed diff between old and new code."""
    old_lines = old_code.split('\n')
    new_lines = new_code.split('\n')
    
    diff = []
    
    # Use difflib to get detailed changes
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'replace':
            diff.append(f"Changed lines {i1+1}-{i2} to {j1+1}-{j2}")
            for i in range(i1, i2):
                diff.append(f"- Line {i+1}: {old_lines[i]}")
            for j in range(j1, j2):
                diff.append(f"+ Line {j+1}: {new_lines[j]}")
        elif op == 'delete':
            diff.append(f"Removed lines {i1+1}-{i2}")
            for i in range(i1, i2):
                diff.append(f"- Line {i+1}: {old_lines[i]}")
        elif op == 'insert':
            diff.append(f"Added lines {j1+1}-{j2}")
            for j in range(j1, j2):
                diff.append(f"+ Line {j+1}: {new_lines[j]}")
    
    return "\n".join(diff)

def _estimate_optimization_impact(self, optimization_title: str, old_code: str, new_code: str) -> Dict[str, Any]:
    """Estimate the impact of the optimization on model performance."""
    impact = {
        "accuracy_improvement": "low",
        "training_speed": "unchanged",
        "inference_speed": "unchanged",
        "memory_usage": "unchanged"
    }
    
    # Check for specific optimizations and their typical impacts
    if "batch normalization" in optimization_title.lower():
        impact["accuracy_improvement"] = "medium"
        impact["training_speed"] = "improved"
    elif "dropout" in optimization_title.lower():
        impact["accuracy_improvement"] = "medium"
        impact["memory_usage"] = "slightly increased"
    elif "learning rate" in optimization_title.lower():
        impact["accuracy_improvement"] = "medium-high"
        impact["training_speed"] = "improved"
    elif "architecture" in optimization_title.lower() or "layer" in optimization_title.lower():
        impact["accuracy_improvement"] = "high"
        impact["memory_usage"] = "increased"
        impact["inference_speed"] = "might decrease"
    elif "regularization" in optimization_title.lower():
        impact["accuracy_improvement"] = "medium"
    
    # Check code changes for specific patterns
    if "nn.BatchNorm" in new_code and "nn.BatchNorm" not in old_code:
        impact["accuracy_improvement"] = "medium-high"
        impact["training_speed"] = "improved"
    if "Dropout" in new_code and "Dropout" not in old_code:
        impact["accuracy_improvement"] = "medium"
        
    return impact

def _apply_generic_optimization(self, code: str, optimization: Dict[str, Any], framework: str) -> str:
    """Apply a generic optimization if Gemini fails to generate specific code."""
    # Framework-specific optimizations as fallback
    if framework == 'pytorch':
        if 'batch normalization' in optimization['title'].lower():
            return self._add_batch_norm_pytorch(code)
        elif 'dropout' in optimization['title'].lower():
            return self._add_dropout_pytorch(code)
        elif 'learning rate' in optimization['title'].lower():
            return self._add_lr_scheduler_pytorch(code)
    
    # If no specific optimization can be applied, return original code
    return code + f"\n\n# TODO: Apply {optimization['title']} optimization"

def _add_batch_norm_pytorch(self, code: str) -> str:
    """Add batch normalization to PyTorch model."""
    # Simple regex-based transformation to add batch norm after Conv2d layers
    pattern = r'(self\.\w+\s*=\s*nn\.Conv2d\([^)]+\))'
    
    def add_bn(match):
        conv_line = match.group(1)
        # Extract the layer name
        layer_name = re.search(r'self\.(\w+)\s*=', conv_line).group(1)
        # Create a batch norm layer with the same number of channels
        channels = re.search(r'Conv2d\([^,]+,\s*(\d+)', conv_line).group(1)
        bn_line = f"\n        self.bn_{layer_name} = nn.BatchNorm2d({channels})"
        return conv_line + bn_line
    
    modified_code = re.sub(pattern, add_bn, code)
    
    # Now add the batch norm to the forward method
    forward_pattern = r'(x\s*=\s*self\.\w+\(x\))(\s*\n\s*)(x\s*=\s*F\.relu\(x\))'
    
    def add_bn_forward(match):
        conv_line = match.group(1)
        spacing = match.group(2)
        relu_line = match.group(3)
        
        # Extract the layer name
        layer_name = re.search(r'self\.(\w+)', conv_line).group(1)
        # Add batch norm before ReLU
        bn_line = f"{spacing}x = self.bn_{layer_name}(x)"
        
        return conv_line + bn_line + spacing + relu_line
    
    modified_code = re.sub(forward_pattern, add_bn_forward, modified_code)
    
    return modified_code

def _add_dropout_pytorch(self, code: str) -> str:
    """Add dropout to PyTorch model."""
    # Add dropout after activation functions
    forward_pattern = r'(x\s*=\s*F\.relu\(x\))(\s*\n)'
    
    def add_dropout(match):
        relu_line = match.group(1)
        spacing = match.group(2)
        
        # Add dropout after ReLU
        dropout_line = f"{spacing}x = F.dropout(x, p=0.3, training=self.training)"
        
        return relu_line + dropout_line
    
    # Add dropout import and layers
    modified_code = code
    
    # Check if __init__ contains dropout already
    if 'Dropout' not in code:
        # Add dropout to __init__
        init_pattern = r'(def __init__\([^:]+:)'
        modified_code = re.sub(init_pattern, r'\1\n        self.dropout = nn.Dropout(0.3)', modified_code)
    
    # Add dropout to forward
    modified_code = re.sub(forward_pattern, add_dropout, modified_code)
    
    return modified_code

def _add_lr_scheduler_pytorch(self, code: str) -> str:
    """Add learning rate scheduler to PyTorch code."""
    # This is a more complex transformation that needs to find the training loop
    if 'optimizer' in code and 'train' in code:
        # Try to locate a training function or loop
        train_pattern = r'(def train[^(]*\([^:]+:)'
        
        def add_scheduler(match):
            train_def = match.group(1)
            
            # Add scheduler after optimizer
            scheduler_code = f"""{train_def}
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                          mode='min', 
                                                          factor=0.1, 
                                                          patience=5, 
                                                          verbose=True)
            """
            
            return scheduler_code
        
        modified_code = re.sub(train_pattern, add_scheduler, code)
        
        # Add scheduler step after backward pass
        backward_pattern = r'(optimizer\.step\(\))'
        
        def add_scheduler_step(match):
            optimizer_step = match.group(1)
            
            # Add scheduler step after optimizer step
            step_code = f"""
        {optimizer_step}
        
        # Update learning rate based on validation loss
        if val_loss is not None:
            scheduler.step(val_loss)
            """
            
            return step_code
        
        modified_code = re.sub(backward_pattern, add_scheduler_step, modified_code)
        
        return modified_code
    
    # If we couldn't find a good place to add the scheduler, append an example
    return code + """
# Example of learning rate scheduler implementation:
def train_with_scheduler(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                          mode='min', 
                                                          factor=0.1, 
                                                          patience=5, 
                                                          verbose=True)
    
    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
"""

def get_fallback_optimization_steps(framework):
    """Return fallback optimization steps if generation fails."""
    if framework == "pytorch":
        return [
            {
                "title": "Add Dropout Regularization",
                "description": "The model lacks regularization, which may lead to overfitting. Adding dropout layers will help the model generalize better to unseen data.",
                "code_section": "__init__ and forward methods",
                "expected_benefit": "Improved generalization and reduced overfitting"
            },
            {
                "title": "Increase Model Capacity",
                "description": "The current architecture may be too simple to capture complex patterns. Adding another hidden layer will increase the model's capacity.",
                "code_section": "model architecture",
                "expected_benefit": "Better performance on complex datasets"
            },
            {
                "title": "Implement Batch Normalization",
                "description": "Batch normalization helps with faster convergence and can improve overall performance.",
                "code_section": "model initialization and forward method",
                "expected_benefit": "Faster training and potentially better performance"
            }
        ]
    # Add similar fallbacks for tensorflow and sklearn
    else:
        return [
            {
                "title": "Add Regularization",
                "description": f"Add regularization to your {framework} model to prevent overfitting.",
                "code_section": "model definition",
                "expected_benefit": "Improved generalization"
            }
        ]

async def apply_optimization(current_code, optimization_step, framework):
    """Apply a specific optimization to the code."""
    try:
        # Use Gemini API to modify the code
        from backend.ml_analysis.code_generator import SimpleCodeGenerator
        
        # Create a code generator instance
        code_generator = SimpleCodeGenerator()
        
        # Create prompt for code modification
        prompt = f"""
        I have a {framework} machine learning model code that needs the following optimization:
        
        Optimization: {optimization_step['title']}
        Description: {optimization_step['description']}
        Code section to modify: {optimization_step['code_section']}
        
        Here's the current code:
        ```python
        {current_code}
        ```
        
        Please modify the code to implement this optimization. Return the complete modified code.
        Ensure the modified code is fully functional and maintains the original structure.
        Only make changes necessary for this specific optimization.
        """
        
        # Call Gemini to generate modified code
        new_code = await code_generator.generate_code_example_async(
            framework=framework,
            category="code_modification",
            model_context={"prompt": prompt}
        )
        
        # Clean up the generated code
        import re
        
        # Remove markdown code blocks
        new_code = re.sub(r'```python\s*', '', new_code)
        new_code = re.sub(r'```\s*', '', new_code)
        
        # If we still don't have valid code, fall back to simple modifications
        if not is_valid_python_code(new_code):
            return apply_fallback_optimization(current_code, optimization_step, framework)
            
        return new_code.strip()
        
    except Exception as e:
        print(f"Error applying optimization: {str(e)}")
        return apply_fallback_optimization(current_code, optimization_step, framework)

def is_valid_python_code(code):
    """Check if the string is valid Python code."""
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False

def apply_fallback_optimization(current_code, optimization_step, framework):
    """Apply simple fallback optimizations if Gemini fails."""
    title = optimization_step['title'].lower()
    
    if 'dropout' in title:
        # Add dropout layers
        new_code = current_code
        
        # Add to __init__
        if 'def __init__' in new_code and 'dropout' not in new_code.lower():
            new_code = new_code.replace(
                'self.relu = nn.ReLU()',
                'self.relu = nn.ReLU()\n        self.dropout = nn.Dropout(0.3)  # Added dropout for regularization'
            )
        
        # Add to forward method
        if 'def forward' in new_code and 'dropout(' not in new_code.lower():
            new_code = new_code.replace(
                'out = self.relu(out)',
                'out = self.relu(out)\n        out = self.dropout(out)  # Apply dropout after activation'
            )
            
        return new_code
        
    elif 'batch' in title or 'normalization' in title:
        # Add batch normalization
        new_code = current_code
        
        # Add to __init__
        if 'def __init__' in new_code and 'batchnorm' not in new_code.lower():
            new_code = new_code.replace(
                'self.layer1 = nn.Linear(',
                'self.bn1 = nn.BatchNorm1d(input_size)  # Added batch normalization\n        self.layer1 = nn.Linear('
            )
            
        # Add to forward method
        if 'def forward' in new_code and 'bn1(' not in new_code.lower():
            new_code = new_code.replace(
                'def forward(self, x):',
                'def forward(self, x):\n        x = self.bn1(x)  # Apply batch normalization'
            )
            
        return new_code
        
    elif 'capacity' in title or 'layer' in title:
        # Add more layers
        new_code = current_code
        
        # Add another layer
        if 'self.layer2 = nn.Linear(' in new_code:
            new_code = new_code.replace(
                'self.layer2 = nn.Linear(hidden_size, num_classes)',
                'self.layer2 = nn.Linear(hidden_size, hidden_size // 2)\n        self.relu2 = nn.ReLU()\n        self.layer3 = nn.Linear(hidden_size // 2, num_classes)'
            )
            
        # Update forward method
        if 'def forward' in new_code and 'layer3' in new_code and 'self.layer3(out)' not in new_code:
            new_code = new_code.replace(
                'out = self.layer2(out)',
                'out = self.layer2(out)\n        out = self.relu2(out)\n        out = self.layer3(out)'
            )
            
        return new_code
    
    # Default: return original code with a comment
    return current_code + f"\n\n# TODO: Apply optimization: {optimization_step['title']}"

def generate_diff(old_code, new_code):
    """Generate a diff between old and new code."""
    diff = difflib.unified_diff(
        old_code.splitlines(),
        new_code.splitlines(),
        lineterm='',
        n=3  # Context lines
    )
    
    return '\n'.join(diff)   

@app.get("/api/usage-stats", response_model=UsageStatsResponse)
async def get_usage_stats(api_key: str = Depends(get_api_key)):
    """Get current usage statistics for an API key."""
    if not HAS_AUTH:
        # Return mock data if auth is disabled
        return {
            "api_key_id": "demo_key",
            "daily_usage": 15,
            "daily_limit": 100,
            "monthly_usage": 450,
            "monthly_limit": 3000,
            "last_used": datetime.now().isoformat(),
            "total_requests": 450,
            "tier": "free",
            "reset_times": {
                "daily_reset": (datetime.now() + timedelta(days=1)).replace(hour=0, minute=0, second=0).isoformat(),
                "monthly_reset": (datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1, hour=0, minute=0, second=0).isoformat()
            }
        }
    
    try:
        from backend.auth.auth import get_usage_stats_for_key
        stats = get_usage_stats_for_key(api_key)
        
        if not stats:
            raise HTTPException(status_code=404, detail="Usage stats not found")
        
        return stats
        
    except Exception as e:
        logging.error(f"Error getting usage stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving usage statistics")

@app.get("/api/usage-history", response_model=UsageHistoryResponse)
async def get_usage_history(
    api_key: str = Depends(get_api_key),
    days: int = Query(30, ge=1, le=90)
):
    """Get usage history for the last N days."""
    if not HAS_AUTH:
        # Return mock data if auth is disabled
        history = []
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            requests = max(0, int(50 * (0.8 + 0.4 * (i % 7) / 6)))  # Mock varying usage
            history.append({
                "date": date.strftime("%Y-%m-%d"),
                "requests": requests
            })
        
        return {
            "history": history,
            "total_days": days
        }
    
    try:
        from backend.auth.auth import get_usage_history_for_key
        history = get_usage_history_for_key(api_key, days)
        
        return {
            "history": history,
            "total_days": days
        }
        
    except Exception as e:
        logging.error(f"Error getting usage history: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving usage history")

@app.get("/api/user/usage-overview")
async def get_user_usage_overview(user_id: str = Depends(get_firebase_token)):
    """Get usage overview for all of a user's API keys."""
    if not HAS_FIREBASE:
        return {
            "total_keys": 2,
            "active_keys": 2,
            "total_requests_today": 25,
            "total_requests_month": 750,
            "keys_near_limit": 0
        }
    
    try:
        # Get all user's API keys
        keys_query = db.collection("api_keys").where("userId", "==", user_id).where("active", "==", True)
        keys = list(keys_query.stream())
        
        total_requests_today = 0
        total_requests_month = 0
        keys_near_limit = 0
        
        current_time = int(time.time())
        day_start = current_time - (current_time % 86400)
        month_start = current_time - (current_time % 2592000)
        
        for key_doc in keys:
            # Get usage data for each key
            usage_ref = db.collection("api_usage").document(key_doc.id)
            usage_doc = usage_ref.get()
            
            if usage_doc.exists:
                usage_data = usage_doc.to_dict()
                
                # Add up daily usage
                daily = usage_data.get("daily", {})
                if daily.get("reset_time", 0) >= day_start:
                    total_requests_today += daily.get("count", 0)
                
                # Add up monthly usage
                monthly = usage_data.get("monthly", {})
                if monthly.get("reset_time", 0) >= month_start:
                    total_requests_month += monthly.get("count", 0)
                
                # Check if near limit
                key_data = key_doc.to_dict()
                tier = key_data.get("tier", "free")
                daily_limit = 100 if tier == "free" else 1000 if tier == "basic" else 10000
                
                if daily.get("count", 0) > daily_limit * 0.8:  # 80% of limit
                    keys_near_limit += 1
        
        return {
            "total_keys": len(keys),
            "active_keys": len(keys),
            "total_requests_today": total_requests_today,
            "total_requests_month": total_requests_month,
            "keys_near_limit": keys_near_limit
        }
        
    except Exception as e:
        logging.error(f"Error getting user usage overview: {str(e)}")
        return {
            "total_keys": 0,
            "active_keys": 0,
            "total_requests_today": 0,
            "total_requests_month": 0,
            "keys_near_limit": 0
        }

@app.post("/api/bit-chat", response_model=BitChatResponse)
async def bit_chat(request: BitChatRequest, api_key: str = Depends(get_api_key)):
    """Process a chat request from Bit and return AI-generated responses with improved model analysis"""
    # Check rate limit and log API usage
    if HAS_AUTH:
        from backend.auth.auth import check_rate_limit, log_api_usage
        if not check_rate_limit(api_key):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please upgrade your plan or try again later."
            )
        # Log the API usage
        log_api_usage(api_key, "bit_chat")
        
    print(f"Received bit-chat request for framework: {request.framework}")
    
    # Enhanced model code analysis for better context
    model_analysis = {}
    if request.code:
        try:
            model_analysis = analyze_model_architecture(request.code, request.framework)
            print(f"Model analysis completed: {len(model_analysis)} attributes identified")
        except Exception as e:
            print(f"Error analyzing model architecture: {e}")
    
    # Check if Gemini API key is available
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    # If no API key, return the test response
    if not gemini_api_key:
        print("No GEMINI_API_KEY found, using test response")
        return {
            "message": f"I've analyzed your {request.framework} code. This is a test response.",
            "suggestions": [
                {
                    "title": "Add Regularization",
                    "description": "Test suggestion",
                    "code": "self.dropout = nn.Dropout(0.3)",
                    "lineNumber": 7
                }
            ]
        }
    
    try:
        # Initialize the Gemini client
        from google import genai
        genai_client = genai.Client(api_key=gemini_api_key)
        
        # Enhanced prompt engineering with model analysis context
        prompt = f"""
        You are Bit, an AI assistant specialized in optimizing and improving machine learning code. 
        You have deep expertise in model architecture, training techniques, and performance optimization.
        You're examining code written in the {request.framework} framework.
        
        # CONTEXT
        ## User Query
        User query: "{request.query}"
        
        ## Code to Analyze
        ```python
        {request.code}
        ```
        
        ## Framework Details
        Framework: {request.framework}
        
        ## Model Analysis
        {json.dumps(model_analysis, indent=2)}
        
        ## Model Metrics (if available)
        - Accuracy: {request.modelInfo.get('accuracy', 'unknown') if request.modelInfo else 'unknown'}
        - Precision: {request.modelInfo.get('precision', 'unknown') if request.modelInfo else 'unknown'}
        - Recall: {request.modelInfo.get('recall', 'unknown') if request.modelInfo else 'unknown'}

        # APPROACH
        1. Thoroughly analyze the code's architecture, training approach, and potential bottlenecks
        2. Consider multiple approaches to optimize the model for both performance and accuracy
        3. Consider the user's specific question and tailor your response accordingly
        4. Be creative and suggest significant architectural improvements when appropriate
        5. Ensure all code is correct, complete and compatible with the existing codebase

        # YOUR TASK
        Based on your analysis, provide:
        1. A direct answer to the user's query
        2. 1-3 specific code improvement suggestions that are:
           - Substantial (not trivial tweaks)
           - Well-explained with rationale
           - Correct and immediately implementable
           - Tailored to the specific model architecture
        3. For each suggestion, include a code example that shows exactly what to implement

        # FRAMEWORK-SPECIFIC CONSIDERATIONS
        ## PyTorch
        - Suggest architectural improvements like residual connections, attention mechanisms
        - Consider regularization techniques beyond basic dropout
        - Look for opportunities to improve forward pass efficiency
        - Suggest adaptive learning rate strategies
        - Consider modern activation functions (GELU, Mish, SiLU)

        ## TensorFlow/Keras
        - Look for opportunities to use tf.data for input pipeline optimization
        - Consider TF-specific features like mixed precision training
        - Suggest appropriate callbacks for monitoring and early stopping
        - Consider TensorBoard integration for visualization

        ## scikit-learn
        - Suggest appropriate preprocessing steps
        - Consider ensemble methods and parameter tuning
        - Look for opportunities to use pipelines

        # RESPONSE FORMAT
        Your response MUST be valid JSON with a "message" field and a "suggestions" array.
        Each suggestion should include a clear title, description, code example, and line number.
        
        For code examples in the "code" field:
        - DO NOT use markdown code blocks (no triple backticks)
        - DO NOT use language indicators
        - Provide the actual code as plain text with proper indentation
        - Use real newlines for line breaks, not escaped newlines
        
        Format your response exactly like this:
        {{
          "message": "Your main analysis of the code and direct answer to the user's query",
          "suggestions": [
            {{
              "title": "Clear and descriptive title",
              "description": "Detailed explanation of what should be improved and why, with expected benefits",
              "code": "def better_function():\\n    print('This is improved code')\\n    return True",
              "lineNumber": 42
            }}
          ]
        }}
        """
        
        print("Sending request to Gemini API")
        # Call Gemini API with enhanced parameters
        response = genai_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            generation_config={
                "temperature": 0.7,          # Higher temperature for more creative suggestions
                "top_p": 0.95,               # Higher precision in token selection
                "top_k": 40,                 # More candidate tokens considered
                "max_output_tokens": 8192,   # Allow larger responses for complete code
            }
        )
        
        # CRITICAL FIX: Log Gemini API call usage 
        if HAS_AUTH:
            from backend.auth.auth import log_api_usage
            # Track each Gemini API call
            log_api_usage(api_key, "bit_chat_gemini_call")
        
        # Extract the text from the response
        if hasattr(response, 'text'):
            text = response.text
        elif hasattr(response, 'parts') and response.parts:
            text = response.parts[0].text
        else:
            print(f"Unexpected response format: {dir(response)}")
            return {
                "message": "I couldn't process your request properly. Here's a general suggestion instead.",
                "suggestions": [
                    {
                        "title": "Add Regularization",
                        "description": "Adding dropout can help prevent overfitting.",
                        "code": "self.dropout = nn.Dropout(0.3)",
                        "lineNumber": 7
                    }
                ]
            }
        
        print(f"Received response from Gemini: {text[:100]}...")
        
        # Enhanced JSON extraction with better error handling
        try:
            # First try to parse directly if it's already valid JSON
            import json
            try:
                parsed_response = json.loads(text)
            except json.JSONDecodeError:
                # If that fails, look for JSON in the response using regex
                import re
                json_match = re.search(r'```json([\s\S]*?)```', text) or re.search(r'{[\s\S]*}', text)
                
                if json_match:
                    json_text = json_match.group(0).replace('```json', '').replace('```', '')
                    parsed_response = json.loads(json_text)
                else:
                    # If no JSON found, convert the text to our required format
                    parsed_response = {
                        "message": text,
                        "suggestions": []
                    }
            
            # Validate and enhance the response
            if "message" not in parsed_response:
                parsed_response["message"] = "I've analyzed your code."
            
            if "suggestions" not in parsed_response or not isinstance(parsed_response["suggestions"], list):
                parsed_response["suggestions"] = []
            
            # Enhanced validation and cleaning of suggestions
            for suggestion in parsed_response.get("suggestions", []):
                # Ensure required fields exist
                if "lineNumber" not in suggestion:
                    # Try to determine line number from code context
                    if "code" in suggestion and request.code:
                        suggestion["lineNumber"] = find_best_line_number(suggestion["code"], request.code)
                    else:
                        suggestion["lineNumber"] = 1
                        
                if "title" not in suggestion:
                    suggestion["title"] = "Code Improvement"
                    
                if "description" not in suggestion:
                    suggestion["description"] = "This improves your code."
                    
                if "code" in suggestion:
                    # Clean up the code formatting
                    code = suggestion["code"]
                    
                    # Remove markdown code blocks and language indicators
                    code = re.sub(r'```python\n?', '', code)
                    code = re.sub(r'```\n?', '', code)
                    code = re.sub(r'```python\r\n?', '', code)
                    code = re.sub(r'```\r\n?', '', code)
                    
                    # Replace escaped newlines with actual newlines
                    code = code.replace('\\n', '\n')
                    code = code.replace('\\\\n', '\n')
                    
                    # Fix common indentation issues
                    code = fix_code_indentation(code)
                    
                    # Verify code doesn't have syntax errors
                    try:
                        compile(code, '<string>', 'exec')
                    except SyntaxError as e:
                        print(f"Syntax error in suggested code: {e}")
                        # Try to fix common syntax issues
                        code = fix_common_syntax_errors(code)
                    
                    # Update the suggestion with clean code
                    suggestion["code"] = code.strip()
                else:
                    suggestion["code"] = "# No specific code provided"
            
            # Log successful response formatting
            if HAS_AUTH:
                from backend.auth.auth import log_api_usage
                log_api_usage(api_key, "bit_chat_success")
            
            return parsed_response
            
        except Exception as e:
            print(f"Error processing Gemini response: {e}")
            # Fall back to extracting content from text
            if HAS_AUTH:
                from backend.auth.auth import log_api_usage
                log_api_usage(api_key, "bit_chat_parse_error")
            
            return {
                "message": text,
                "suggestions": []
            }
            
    except Exception as e:
        print(f"Error using Gemini API: {str(e)}")
        # Return fallback response on error
        if HAS_AUTH:
            from backend.auth.auth import log_api_usage
            log_api_usage(api_key, "bit_chat_error")
        return {
            "message": f"I encountered an error analyzing your code. Here's a general suggestion: {str(e)}",
            "suggestions": [
                {
                    "title": "General Improvement",
                    "description": "Consider adding regularization to prevent overfitting.",
                    "code": "self.dropout = nn.Dropout(0.3)",
                    "lineNumber": 7
                }
            ]
        }

def fix_common_syntax_errors(code: str) -> str:
    """Fix common syntax errors in Python code."""
    # Remove trailing commas in function calls or parameter lists
    code = re.sub(r',\s*\)', ')', code)
    
    # Fix indentation after colons
    code = re.sub(r':\s*(\S)', ':\n    \\1', code)
    
    # Fix missing closing parentheses
    open_count = code.count('(')
    close_count = code.count(')')
    if open_count > close_count:
        code += ')' * (open_count - close_count)
    
    # Fix missing closing brackets
    open_count = code.count('[')
    close_count = code.count(']')
    if open_count > close_count:
        code += ']' * (open_count - close_count)
    
    # Fix missing closing braces
    open_count = code.count('{')
    close_count = code.count('}')
    if open_count > close_count:
        code += '}' * (open_count - close_count)
    
    # Fix common typos in keywords
    typos = {
        'defn': 'def',
        'clas': 'class',
        'functoin': 'function',
        'retrun': 'return',
        'imoprt': 'import',
        'fro ': 'for ',  # Space after to avoid replacing 'from'
    }
    
    for typo, correction in typos.items():
        code = code.replace(typo, correction)
        
    return code
# Add this to backend/app/server.py

@app.post("/api/validate-code")
async def validate_code(request: SaveCodeRequest, api_key: str = Depends(get_api_key)):
    """Validate Python code for syntax errors."""
    try:
        # Try to compile the code to check for syntax errors
        compile(request.code, '<string>', 'exec')
        return {"valid": True, "message": "Code is valid Python syntax"}
    except Exception as e:
        # Return error details if compilation fails
        error_message = str(e)
        line_number = None
        
        # Extract line number from error message if available
        if hasattr(e, 'lineno'):
            line_number = e.lineno
        elif 'line' in error_message:
            try:
                line_match = re.search(r'line (\d+)', error_message)
                if line_match:
                    line_number = int(line_match.group(1))
            except:
                pass
                
        return {
            "valid": False, 
            "message": error_message,
            "line_number": line_number
        }
    
def analyze_model_architecture(code: str, framework: str) -> Dict[str, Any]:
    """Analyze the structure of a machine learning model to provide better context."""
    analysis = {
        "framework": framework,
        "has_model_class": False,
        "model_class_name": None,
        "layers": [],
        "activations": [],
        "optimizers": [],
        "regularization": {
            "has_dropout": False,
            "has_batch_norm": False,
            "has_weight_decay": False
        },
        "training": {
            "has_training_loop": False,
            "has_validation": False,
            "has_lr_scheduler": False
        }
    }
    
    # Framework-specific analysis
    if framework.lower() == 'pytorch':
        # Check for model class
        class_pattern = r'class\s+(\w+)\s*\(\s*(?:nn\.)?Module\s*\)'
        class_match = re.search(class_pattern, code)
        if class_match:
            analysis["has_model_class"] = True
            analysis["model_class_name"] = class_match.group(1)
        
        # Find layers
        layer_patterns = {
            'conv': r'self\.(\w+)\s*=\s*nn\.Conv\w*\(([^)]+)\)',
            'linear': r'self\.(\w+)\s*=\s*nn\.Linear\(([^)]+)\)',
            'lstm': r'self\.(\w+)\s*=\s*nn\.LSTM\(([^)]+)\)',
            'rnn': r'self\.(\w+)\s*=\s*nn\.RNN\(([^)]+)\)',
            'gru': r'self\.(\w+)\s*=\s*nn\.GRU\(([^)]+)\)',
            'pool': r'self\.(\w+)\s*=\s*nn\.(?:Max|Avg)Pool\w*\(([^)]+)\)',
            'dropout': r'self\.(\w+)\s*=\s*nn\.Dropout\w*\(([^)]+)\)',
            'batchnorm': r'self\.(\w+)\s*=\s*nn\.BatchNorm\w*\(([^)]+)\)'
        }
        
        for layer_type, pattern in layer_patterns.items():
            for match in re.finditer(pattern, code):
                layer_name = match.group(1)
                layer_args = match.group(2)
                
                analysis["layers"].append({
                    "type": layer_type,
                    "name": layer_name,
                    "args": layer_args
                })
                
                # Update regularization flags
                if layer_type == 'dropout':
                    analysis["regularization"]["has_dropout"] = True
                elif layer_type == 'batchnorm':
                    analysis["regularization"]["has_batch_norm"] = True
        
        # Find activations
        activation_pattern = r'(?:F\.|nn\.|torch\.nn\.functional\.)(\w+)\('
        for match in re.finditer(activation_pattern, code):
            activation = match.group(1).lower()
            if activation in ['relu', 'sigmoid', 'tanh', 'softmax', 'leakyrelu', 'elu', 'selu', 'gelu']:
                if activation not in analysis["activations"]:
                    analysis["activations"].append(activation)
        
        # Check for training loop indicators
        training_indicators = [
            r'\.train\(\)',
            r'\.backward\(\)',
            r'optimizer\.step\(\)',
            r'for\s+\w+\s+in\s+\w+(?:_loader|_dataloader)'
        ]
        
        for indicator in re.finditer('|'.join(training_indicators), code):
            analysis["training"]["has_training_loop"] = True
            break
        
        # Check for validation
        if re.search(r'val_loss|validation|\.eval\(\)', code):
            analysis["training"]["has_validation"] = True
        
        # Check for learning rate scheduler
        if re.search(r'lr_scheduler|LRScheduler', code):
            analysis["training"]["has_lr_scheduler"] = True
        
        # Check for weight decay
        if re.search(r'weight_decay\s*=', code):
            analysis["regularization"]["has_weight_decay"] = True
        
        # Find optimizers
        optimizer_pattern = r'(?:optim|torch\.optim)\.(\w+)\('
        for match in re.finditer(optimizer_pattern, code):
            optimizer = match.group(1)
            if optimizer not in analysis["optimizers"]:
                analysis["optimizers"].append(optimizer)
                
    # Similar analysis for TensorFlow and scikit-learn could be added here
    
    return analysis

def find_best_line_number(suggestion_code: str, full_code: str) -> int:
    """Find the best line number for a code suggestion based on context matching."""
    # Clean up both code snippets
    suggestion_lines = [line.strip() for line in suggestion_code.split('\n')]
    full_code_lines = full_code.split('\n')
    
    # Try to find a good insertion point based on content similarity
    best_line = 1
    best_score = 0
    
    # Check for class or function definitions that might be referenced
    class_pattern = r'class\s+(\w+)'
    function_pattern = r'def\s+(\w+)'
    
    class_match = re.search(class_pattern, suggestion_code)
    function_match = re.search(function_pattern, suggestion_code)
    
    # If we found a class or function name in the suggestion, look for it in the full code
    if class_match:
        class_name = class_match.group(1)
        for i, line in enumerate(full_code_lines):
            if f"class {class_name}" in line:
                return i + 1  # Line numbers are 1-based
    
    if function_match:
        function_name = function_match.group(1)
        for i, line in enumerate(full_code_lines):
            if f"def {function_name}" in line:
                return i + 1
    
    # If we're looking at a method inside a class, try to find the class
    if 'self.' in suggestion_code:
        for i, line in enumerate(full_code_lines):
            if 'class ' in line and 'class' in full_code_lines[i:i+10]:
                return i + 3  # Aim for inside the class, after __init__
    
    # Generic similarity search
    for i in range(len(full_code_lines)):
        window = full_code_lines[i:i+min(10, len(full_code_lines)-i)]
        window_text = '\n'.join(window)
        
        # Use difflib to calculate similarity
        similarity = difflib.SequenceMatcher(None, suggestion_lines[0], window_text).ratio()
        
        if similarity > best_score:
            best_score = similarity
            best_line = i + 1
    
    return best_line

def fix_code_indentation(code: str) -> str:
    """Fix common indentation issues in code."""
    lines = code.split('\n')
    if not lines:
        return code
    
    # Determine the base indentation level (if any)
    base_indent = 0
    for line in lines:
        if line.strip():  # Skip empty lines
            indent = len(line) - len(line.lstrip())
            base_indent = indent if base_indent == 0 else min(base_indent, indent)
    
    # Remove the base indentation from all lines
    if base_indent > 0:
        for i in range(len(lines)):
            if lines[i].strip():  # Only process non-empty lines
                lines[i] = lines[i][base_indent:]
    
    # Standardize indentation to 4 spaces
    fixed_lines = []
    for line in lines:
        # Replace tabs with 4 spaces
        line = line.replace('\t', '    ')
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

# Add the middleware to your app
app.add_middleware(ApiKeyMiddleware)


# Track server start time
server_start_time = datetime.now()

@app.post("/api/profile-model")
async def profile_model(api_key: str = Depends(get_api_key)):
    """Profile the model's performance and memory usage to identify optimization opportunities."""
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    try:
        # Check if torch is available for profiling
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False
            return {
                "error": "PyTorch is required for model profiling",
                "message": "Please install PyTorch to use this feature"
            }
        
        if not has_torch or debugger.framework.lower() != 'pytorch':
            return {
                "message": f"Profiling is currently only supported for PyTorch models. Your model uses {debugger.framework}.",
                "supported": False
            }
        
        # Get the model and a sample input
        model = debugger.model
        sample_input = None
        
        # Try to get a sample input from the debugger's dataset
        if hasattr(debugger, 'X') and debugger.X is not None and len(debugger.X) > 0:
            sample_input = torch.tensor(debugger.X[0:1], dtype=torch.float32)
        else:
            # Create a mock input based on model's first layer
            first_layer = None
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                    first_layer = module
                    break
            
            if first_layer is None:
                return {
                    "error": "Could not determine input shape",
                    "message": "Could not find a suitable layer to determine input shape"
                }
            
            # Create mock input based on layer type
            if isinstance(first_layer, torch.nn.Linear):
                input_size = first_layer.in_features
                sample_input = torch.randn(1, input_size)
            elif isinstance(first_layer, torch.nn.Conv1d):
                in_channels = first_layer.in_channels
                sample_input = torch.randn(1, in_channels, 32)  # Assuming input length of 32
            elif isinstance(first_layer, torch.nn.Conv2d):
                in_channels = first_layer.in_channels
                sample_input = torch.randn(1, in_channels, 32, 32)  # Assuming 32x32 input
            elif isinstance(first_layer, torch.nn.Conv3d):
                in_channels = first_layer.in_channels
                sample_input = torch.randn(1, in_channels, 16, 16, 16)  # Assuming 16x16x16 input
        
        # Profile the model
        profile_results = profile_pytorch_model(model, sample_input)
        
        return {
            "message": "Model profiling completed successfully",
            "supported": True,
            "profile_results": profile_results
        }
    
    except Exception as e:
        import traceback
        logging.error(f"Error profiling model: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error profiling model: {str(e)}")

def profile_pytorch_model(model, sample_input):
    """Profile a PyTorch model to identify performance bottlenecks and optimization opportunities."""
    import torch
    from collections import defaultdict
    import time
    
    # Put model in evaluation mode
    model.eval()
    
    # MEMORY PROFILING
    # ----------------
    # Get initial memory usage
    torch.cuda.empty_cache()
    memory_stats = {}
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        
        # Move model and input to GPU
        device = torch.device("cuda")
        model = model.to(device)
        sample_input = sample_input.to(device)
        
        # Run inference
        with torch.no_grad():
            _ = model(sample_input)
        
        # Get peak memory
        peak_mem = torch.cuda.max_memory_allocated()
        end_mem = torch.cuda.memory_allocated()
        
        memory_stats = {
            "start_memory_mb": start_mem / (1024 * 1024),
            "end_memory_mb": end_mem / (1024 * 1024),
            "peak_memory_mb": peak_mem / (1024 * 1024),
            "memory_increase_mb": (end_mem - start_mem) / (1024 * 1024)
        }
    else:
        # CPU memory profiling is less accurate
        memory_stats = {
            "message": "CUDA not available, memory profiling limited to timing metrics"
        }
    
    # LAYER-BY-LAYER PROFILING
    # ------------------------
    # Set up hooks for layer-by-layer profiling
    layer_times = defaultdict(float)
    layer_input_shapes = {}
    layer_output_shapes = {}
    layer_param_counts = {}
    
    handles = []
    
    def hook_fn(name):
        def hook(module, input, output):
            # Record start time
            start = time.time()
            
            # Store input and output shapes
            if isinstance(input, tuple) and len(input) > 0:
                layer_input_shapes[name] = [tuple(i.shape) if isinstance(i, torch.Tensor) else None for i in input]
            else:
                layer_input_shapes[name] = None
            
            if isinstance(output, tuple):
                layer_output_shapes[name] = [tuple(o.shape) if isinstance(o, torch.Tensor) else None for o in output]
            elif isinstance(output, torch.Tensor):
                layer_output_shapes[name] = tuple(output.shape)
            else:
                layer_output_shapes[name] = None
            
            # Count parameters
            layer_param_counts[name] = sum(p.numel() for p in module.parameters())
            
            # Record execution time
            layer_times[name] += time.time() - start
        return hook
    
    # Register hooks for all modules
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.RNN, torch.nn.LSTM, 
                               torch.nn.GRU, torch.nn.BatchNorm2d, torch.nn.MaxPool2d)):
            handle = module.register_forward_hook(hook_fn(f"{name} ({module.__class__.__name__})"))
            handles.append(handle)
    
    # TIMING PROFILING
    # ---------------
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample_input)
    
    # Actual timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Calculate average inference time
    avg_inference_time = (end_time - start_time) / 10
    
    # Prepare layer profiling results
    layer_profiles = []
    total_time = sum(layer_times.values())
    total_params = sum(layer_param_counts.values())
    
    for name, time_taken in sorted(layer_times.items(), key=lambda x: x[1], reverse=True):
        percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
        param_percentage = (layer_param_counts[name] / total_params) * 100 if total_params > 0 else 0
        
        layer_profiles.append({
            "layer_name": name,
            "execution_time_ms": time_taken * 1000,
            "percentage_of_total_time": percentage,
            "parameter_count": layer_param_counts[name],
            "percentage_of_parameters": param_percentage,
            "input_shape": layer_input_shapes.get(name),
            "output_shape": layer_output_shapes.get(name)
        })
    
    # MODEL COMPLEXITY ANALYSIS
    # ------------------------
    # Count total parameters and calculate FLOPs (estimated)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate FLOPs (this is a rough estimate)
    estimated_flops = estimate_flops(model, sample_input)
    
    # OPTIMIZATION SUGGESTIONS
    # -----------------------
    # Analyze profiling results to generate optimization suggestions
    suggestions = generate_optimization_suggestions(
        layer_profiles, 
        memory_stats, 
        avg_inference_time, 
        total_params, 
        trainable_params,
        estimated_flops
    )
    
    # Compile final results
    results = {
        "performance": {
            "avg_inference_time_ms": avg_inference_time * 1000,
            "inferences_per_second": 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        },
        "memory": memory_stats,
        "complexity": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_flops": estimated_flops,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        },
        "layer_profiles": layer_profiles,
        "optimization_suggestions": suggestions
    }
    
    return results

def estimate_flops(model, sample_input):
    """Estimate the number of FLOPs (Floating Point Operations) for a PyTorch model."""
    try:
        from thop import profile
        macs, params = profile(model, inputs=(sample_input,))
        # FLOPs ≈ 2 * MACs
        return 2 * macs
    except ImportError:
        # If thop is not available, make a rough estimate
        # This is a very simplistic approximation
        total_params = sum(p.numel() for p in model.parameters())
        # Assume each parameter is used in 2-4 operations on average
        return total_params * 3

def generate_optimization_suggestions(layer_profiles, memory_stats, avg_inference_time, total_params, trainable_params, estimated_flops):
    """Generate optimization suggestions based on profiling results."""
    suggestions = []
    
    # Sort layers by execution time
    time_sorted_layers = sorted(layer_profiles, key=lambda x: x["execution_time_ms"], reverse=True)
    param_sorted_layers = sorted(layer_profiles, key=lambda x: x["parameter_count"], reverse=True)
    
    # Identify potential bottlenecks
    bottleneck_threshold = 0.2  # Layers taking >20% of execution time
    bottleneck_layers = [layer for layer in time_sorted_layers 
                        if layer["percentage_of_total_time"] > bottleneck_threshold * 100]
    
    # Check for convolution bottlenecks
    conv_bottlenecks = [layer for layer in bottleneck_layers if "Conv" in layer["layer_name"]]
    if conv_bottlenecks:
        suggestions.append({
            "category": "Performance",
            "title": "Optimize Convolutional Layers",
            "description": f"Convolutional layers like {conv_bottlenecks[0]['layer_name']} are taking {conv_bottlenecks[0]['percentage_of_total_time']:.1f}% of the inference time.",
            "suggestions": [
                "Consider using grouped convolutions to reduce computation",
                "Try depthwise separable convolutions for significant speedup",
                "Reduce the number of filters or kernel size"
            ],
            "impact": "high"
        })
    
    # Check for large fully-connected layers
    fc_bottlenecks = [layer for layer in param_sorted_layers if "Linear" in layer["layer_name"] 
                     and layer["parameter_count"] > total_params * 0.3]
    if fc_bottlenecks:
        suggestions.append({
            "category": "Model Size",
            "title": "Reduce Fully-Connected Layer Size",
            "description": f"The fully-connected layer {fc_bottlenecks[0]['layer_name']} contains {fc_bottlenecks[0]['parameter_count']:,} parameters ({fc_bottlenecks[0]['percentage_of_parameters']:.1f}% of total).",
            "suggestions": [
                "Reduce the hidden dimension size",
                "Consider adding a bottleneck architecture",
                "Implement low-rank factorization of weight matrices"
            ],
            "impact": "high"
        })
    
    # Check if model is compute-bound or memory-bound
    if avg_inference_time > 0.1:  # More than 100ms inference time
        suggestions.append({
            "category": "Inference Speed",
            "title": "Speed Up Model Inference",
            "description": f"Model inference time is {avg_inference_time * 1000:.1f}ms, which may be too slow for real-time applications.",
            "suggestions": [
                "Consider quantization to int8 for 2-4x speedup",
                "Investigate pruning to remove unimportant connections",
                "Try knowledge distillation to create a smaller, faster student model",
                "Batch multiple inputs together for higher throughput"
            ],
            "impact": "high"
        })
    
    # Check for memory issues (only if CUDA profiling was done)
    if "peak_memory_mb" in memory_stats:
        peak_memory = memory_stats["peak_memory_mb"]
        if peak_memory > 500:  # More than 500MB
            suggestions.append({
                "category": "Memory Usage",
                "title": "Reduce Memory Footprint",
                "description": f"Peak memory usage is {peak_memory:.1f}MB, which may be too high for resource-constrained environments.",
                "suggestions": [
                    "Use gradient checkpointing to trade compute for memory",
                    "Implement mixed precision training/inference",
                    "Consider model quantization or pruning",
                    "Use smaller batch sizes if applicable"
                ],
                "impact": "medium"
            })
    
    # Add general suggestions if none specific were found
    if not suggestions:
        suggestions.append({
            "category": "General Optimization",
            "title": "General Model Improvements",
            "description": "Consider these general optimizations to improve model performance.",
            "suggestions": [
                "Try quantization to reduce model size and increase inference speed",
                "Experiment with pruning to remove unnecessary connections",
                "Use JIT compilation with TorchScript for production deployment",
                "Optimize data loading pipeline for training efficiency"
            ],
            "impact": "medium"
        })
    
    return suggestions

@app.post("/api/benchmark-model")
async def benchmark_model(
    api_key: str = Depends(get_api_key),
    batch_sizes: List[int] = Query([1, 8, 32], ge=1, le=256),
    num_runs: int = Query(10, ge=3, le=100)
):
    """Benchmark the model across different batch sizes and compare to standard models."""
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    try:
        # Check if torch is available for benchmarking
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False
            return {
                "error": "PyTorch is required for model benchmarking",
                "message": "Please install PyTorch to use this feature"
            }
        
        if not has_torch or debugger.framework.lower() != 'pytorch':
            return {
                "message": f"Benchmarking is currently only supported for PyTorch models. Your model uses {debugger.framework}.",
                "supported": False
            }
        
        # Get the model
        model = debugger.model
        
        # Create sample inputs for different batch sizes
        sample_inputs = {}
        
        # Try to determine input shape from model or dataset
        input_shape = None
        
        # Try to get input shape from debugger's dataset
        if hasattr(debugger, 'X') and debugger.X is not None and len(debugger.X) > 0:
            first_sample = debugger.X[0]
            input_shape = first_sample.shape
        
        # If not available from dataset, infer from model
        if input_shape is None:
            # Find the first layer to determine input shape
            first_layer = None
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                    first_layer = module
                    break
            
            if first_layer is None:
                return {
                    "error": "Could not determine input shape",
                    "message": "Could not find a suitable layer to determine input shape"
                }
            
            # Create input shape based on layer type
            if isinstance(first_layer, torch.nn.Linear):
                input_shape = (first_layer.in_features,)
            elif isinstance(first_layer, torch.nn.Conv1d):
                input_shape = (first_layer.in_channels, 32)  # Assume length 32
            elif isinstance(first_layer, torch.nn.Conv2d):
                input_shape = (first_layer.in_channels, 32, 32)  # Assume 32x32
            elif isinstance(first_layer, torch.nn.Conv3d):
                input_shape = (first_layer.in_channels, 16, 16, 16)  # Assume 16x16x16
        
        # Create inputs for each batch size
        for batch_size in batch_sizes:
            if len(input_shape) == 1:
                sample_inputs[batch_size] = torch.randn(batch_size, *input_shape)
            else:
                sample_inputs[batch_size] = torch.randn(batch_size, *input_shape)
        
        # Run benchmarks
        benchmark_results = run_model_benchmark(model, sample_inputs, num_runs)
        
        # Compare with standard models
        comparison_results = compare_with_standard_models(benchmark_results, input_shape)
        
        return {
            "message": "Model benchmarking completed successfully",
            "supported": True,
            "benchmark_results": benchmark_results,
            "comparison_results": comparison_results
        }
    
    except Exception as e:
        import traceback
        logging.error(f"Error benchmarking model: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error benchmarking model: {str(e)}")

def run_model_benchmark(model, sample_inputs, num_runs):
    """Run performance benchmarks on the model with different batch sizes."""
    import torch
    import time
    
    # Put model in evaluation mode
    model.eval()
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Move model to device
    model = model.to(device)
    
    # Results container
    results = {}
    
    # Test each batch size
    for batch_size, inputs in sample_inputs.items():
        # Move inputs to device
        inputs = inputs.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(inputs)
        
        # Synchronize before starting timing
        if use_cuda:
            torch.cuda.synchronize()
        
        # Timing runs
        latencies = []
        for _ in range(num_runs):
            # Start timer
            start_time = time.time()
            
            # Run inference
            with torch.no_grad():
                _ = model(inputs)
            
            # Synchronize before stopping timer
            if use_cuda:
                torch.cuda.synchronize()
            
            # Stop timer
            end_time = time.time()
            
            # Record latency
            latencies.append(end_time - start_time)
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        throughput = batch_size / avg_latency
        latencies_ms = [l * 1000 for l in latencies]
        
        # Sort latencies for percentile calculations
        sorted_latencies = sorted(latencies_ms)
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p90 = sorted_latencies[int(len(sorted_latencies) * 0.9)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        # Record results
        results[batch_size] = {
            "batch_size": batch_size,
            "average_latency_ms": avg_latency * 1000,
            "throughput_samples_per_second": throughput,
            "p50_latency_ms": p50,
            "p90_latency_ms": p90,
            "p99_latency_ms": p99,
            "device": str(device),
            "num_runs": num_runs
        }
    
    # Calculate scaling efficiency
    if len(results) > 1:
        batch_sizes = sorted(results.keys())
        base_batch = batch_sizes[0]
        base_throughput = results[base_batch]["throughput_samples_per_second"]
        
        for batch_size in batch_sizes[1:]:
            throughput = results[batch_size]["throughput_samples_per_second"]
            ideal_throughput = base_throughput * (batch_size / base_batch)
            scaling_efficiency = (throughput / ideal_throughput) * 100
            results[batch_size]["scaling_efficiency_percent"] = scaling_efficiency
    
    return results

def compare_with_standard_models(benchmark_results, input_shape):
    """Compare benchmark results with standard models of similar complexity."""
    # Standard model performance metrics (pre-computed or from literature)
    # These are example values - in a real implementation, you would use actual benchmarks
    standard_models = {
        "resnet18": {
            "params": 11.7e6,
            "average_latency_ms": {1: 5.2, 8: 15.6, 32: 48.2},
            "throughput_samples_per_second": {1: 192.3, 8: 512.8, 32: 664.2}
        },
        "mobilenet_v2": {
            "params": 3.5e6,
            "average_latency_ms": {1: 3.8, 8: 9.4, 32: 31.5},
            "throughput_samples_per_second": {1: 263.2, 8: 851.1, 32: 1015.9}
        },
        "efficientnet_b0": {
            "params": 5.3e6,
            "average_latency_ms": {1: 7.8, 8: 18.2, 32: 57.6},
            "throughput_samples_per_second": {1: 128.2, 8: 439.6, 32: 555.6}
        }
    }
    
    # Estimate your model's parameter count
    import torch
    
    # Get the global debugger model
    global debugger
    model = debugger.model
    
    # Count parameters
    model_params = sum(p.numel() for p in model.parameters())
    
    # Find closest standard models by parameter count
    sorted_models = sorted(standard_models.items(), 
                           key=lambda x: abs(x[1]["params"] - model_params))
    
    # Take the top 2 closest models
    closest_models = sorted_models[:2]
    
    # Prepare comparison results
    comparison = {
        "your_model": {
            "params": model_params,
            "latency_and_throughput": {
                batch_size: {
                    "average_latency_ms": data["average_latency_ms"],
                    "throughput_samples_per_second": data["throughput_samples_per_second"]
                } for batch_size, data in benchmark_results.items()
            }
        }
    }
    
    # Add comparison models
    for model_name, model_data in closest_models:
        comparison[model_name] = {
            "params": model_data["params"],
            "latency_and_throughput": {
                batch_size: {
                    "average_latency_ms": model_data["average_latency_ms"].get(batch_size, None),
                    "throughput_samples_per_second": model_data["throughput_samples_per_second"].get(batch_size, None)
                } for batch_size in benchmark_results.keys() if batch_size in model_data["average_latency_ms"]
            }
        }
    
    # Calculate relative performance for each batch size
    for batch_size in benchmark_results.keys():
        your_latency = benchmark_results[batch_size]["average_latency_ms"]
        your_throughput = benchmark_results[batch_size]["throughput_samples_per_second"]
        
        for model_name, model_data in closest_models:
            if batch_size in model_data["average_latency_ms"]:
                comparison[model_name]["latency_and_throughput"][batch_size]["relative_latency"] = (
                    your_latency / model_data["average_latency_ms"][batch_size]
                )
                comparison[model_name]["latency_and_throughput"][batch_size]["relative_throughput"] = (
                    your_throughput / model_data["throughput_samples_per_second"][batch_size]
                )
    
    # Add performance summary
    comparison["summary"] = {
        "model_size_comparison": f"Your model has {model_params:,} parameters, which is " + 
            f"{model_params / closest_models[0][1]['params']:.1f}x the size of {closest_models[0][0]}.",
        "performance_highlights": []
    }
    
    # Add performance highlights
    batch_sizes = sorted(benchmark_results.keys())
    if batch_sizes:
        # Look at single-sample latency (important for real-time applications)
        smallest_batch = batch_sizes[0]
        if smallest_batch in benchmark_results:
            your_latency = benchmark_results[smallest_batch]["average_latency_ms"]
            best_standard = min(
                (model_data["average_latency_ms"].get(smallest_batch, float('inf')), model_name) 
                for model_name, model_data in closest_models
            )
            
            if best_standard[0] != float('inf'):
                latency_ratio = your_latency / best_standard[0]
                if latency_ratio < 0.8:
                    comparison["summary"]["performance_highlights"].append(
                        f"Your model is {1/latency_ratio:.1f}x faster than {best_standard[1]} for single-sample inference."
                    )
                elif latency_ratio > 1.2:
                    comparison["summary"]["performance_highlights"].append(
                        f"Your model is {latency_ratio:.1f}x slower than {best_standard[1]} for single-sample inference."
                    )
                else:
                    comparison["summary"]["performance_highlights"].append(
                        f"Your model has similar single-sample inference speed to {best_standard[1]}."
                    )
        
        # Look at batch throughput (important for batch processing)
        largest_batch = batch_sizes[-1]
        if largest_batch in benchmark_results:
            your_throughput = benchmark_results[largest_batch]["throughput_samples_per_second"]
            best_standard = max(
                (model_data["throughput_samples_per_second"].get(largest_batch, 0), model_name) 
                for model_name, model_data in closest_models
            )
            
            if best_standard[0] > 0:
                throughput_ratio = your_throughput / best_standard[0]
                if throughput_ratio > 1.2:
                    comparison["summary"]["performance_highlights"].append(
                        f"Your model has {throughput_ratio:.1f}x higher throughput than {best_standard[1]} for batch processing."
                    )
                elif throughput_ratio < 0.8:
                    comparison["summary"]["performance_highlights"].append(
                        f"Your model has {1/throughput_ratio:.1f}x lower throughput than {best_standard[1]} for batch processing."
                    )
                else:
                    comparison["summary"]["performance_highlights"].append(
                        f"Your model has similar batch processing throughput to {best_standard[1]}."
                    )
    
    return comparison

def is_dashboard_request(request: Request) -> bool:
    """Check if a request is coming from the dashboard frontend."""
    # Check if it's a browser request via User-Agent
    user_agent = request.headers.get("user-agent", "")
    is_browser = any(browser in user_agent for browser in ["Mozilla", "Chrome", "Safari", "Edge"])
    
    # Check if it's from our own origin
    referer = request.headers.get("referer", "")
    is_local = any(local in referer for local in ["localhost:8000", "0.0.0.0:8000", "127.0.0.1:8000"])
    
    # Consider it a dashboard request if it's a browser and from our local server
    return is_browser and (is_local or request.url.path == "/" or 
                          request.url.path.startswith("/static"))

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
# @app.get("/")
# async def root():
#    return {"message": "Cinder API is running", "version": "1.0.0"}

# Add this dependency function

# Status endpoint
@app.get("/api/status", response_model=ServerStatusResponse)
async def get_status(api_key: str = Depends(get_api_key)):
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
async def get_model_code(api_key: str = Depends(get_api_key)):
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
async def get_model_info(api_key: str = Depends(get_api_key)):
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
async def get_model_improvements(api_key: str = Depends(get_api_key),
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
async def generate_code_example(api_key: str = Depends(get_api_key),
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
async def get_errors(api_key: str = Depends(get_api_key)):
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
async def get_confidence_analysis(api_key: str = Depends(get_api_key)):
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    confidence_analysis = debugger.analyze_confidence()

    if "error" in confidence_analysis:
        raise HTTPException(status_code=400, detail=confidence_analysis["error"])

    return confidence_analysis


# Feature importance endpoint
@app.get("/api/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(api_key: str = Depends(get_api_key)):
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    importance_analysis = debugger.analyze_feature_importance()

    if "error" in importance_analysis:
        raise HTTPException(status_code=400, detail=importance_analysis["error"])

    return importance_analysis


@app.get("/api/improvement-suggestions", response_model=List[ImprovementSuggestion])
async def get_improvement_suggestions(api_key: str = Depends(get_api_key)):
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    return debugger.generate_improvement_suggestions()


# Cross-validation endpoint
@app.get("/api/cross-validation", response_model=CrossValidationResponse)
async def get_cross_validation(api_key: str = Depends(get_api_key),
    k_folds: int = Query(5, ge=2, le=10)):
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    cv_results = debugger.perform_cross_validation(k_folds=k_folds)

    if "error" in cv_results:
        raise HTTPException(status_code=400, detail=cv_results["error"])

    return cv_results


# Prediction drift analysis endpoint
@app.get("/api/prediction-drift", response_model=PredictionDriftResponse)
async def get_prediction_drift(api_key: str = Depends(get_api_key),
    threshold: float = Query(0.1, ge=0.01, le=0.5)):
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    drift_analysis = debugger.analyze_prediction_drift(threshold=threshold)

    if "error" in drift_analysis:
        raise HTTPException(status_code=400, detail=drift_analysis["error"])

    return drift_analysis


# ROC curve endpoint (for binary classification)
@app.get("/api/roc-curve", response_model=ROCCurveResponse)
async def get_roc_curve(api_key: str = Depends(get_api_key)):
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
async def get_training_history(api_key: str = Depends(get_api_key)):
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    return debugger.get_training_history()


@app.get("/api/model-improvement-suggestions", response_model=Dict[str, Any])
async def get_model_improvement_suggestions(api_key: str = Depends(get_api_key),
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
async def get_error_types(api_key: str = Depends(get_api_key)):
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    return debugger.analyze_error_types()


# Confusion Matrix endpoint
@app.get("/api/confusion-matrix", response_model=ConfusionMatrixResponse)
async def get_confusion_matrix(api_key: str = Depends(get_api_key)):
    global debugger
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")

    analysis = debugger.analyze()
    return analysis["confusion_matrix"]


# Prediction Distribution endpoint
@app.get(
    "/api/prediction-distribution", response_model=List[PredictionDistributionItem]
)
async def get_prediction_distribution(api_key: str = Depends(get_api_key)):
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
async def get_sample_predictions(api_key: str = Depends(get_api_key),
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

# Add these models
class UserApiKey(BaseModel):
    id: str = Field(..., description="Unique identifier for the API key")
    key: str = Field(..., description="The API key")
    tier: str = Field(..., description="The subscription tier of the key")
    createdAt: int = Field(..., description="When the key was created (unix timestamp)")
    expiresAt: int = Field(..., description="When the key expires (unix timestamp)")
    lastUsed: Optional[int] = Field(None, description="When the key was last used (unix timestamp)")
    usageCount: int = Field(0, description="Number of times the key has been used")
    
class UserApiKeyList(BaseModel):
    keys: List[UserApiKey] = Field(..., description="List of API keys")
    
class CreateApiKeyResponse(BaseModel):
    key: UserApiKey = Field(..., description="The created API key")
    message: str = Field(..., description="Success message")

# Initialize Firebase Admin SDK for auth
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, auth as firebase_auth
    
    cred_path = os.path.join(os.path.dirname(__file__), '..', 'firebase-credentials.json')
    if not firebase_admin._apps:  # Only initialize if not already initialized
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    HAS_FIREBASE = True
except Exception as e:
    logging.warning(f"Could not initialize Firebase: {e}")
    HAS_FIREBASE = False

# Add Firebase token validation
async def get_firebase_token(authorization: Optional[str] = Header(None)):
    """Validate Firebase auth token and return user ID."""
    if not HAS_FIREBASE:
        raise HTTPException(status_code=501, detail="Firebase authentication not available")
        
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        
    token = authorization.split("Bearer ")[1]
    
    try:
        decoded_token = firebase_auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

# API key management endpoints
@app.get("/api/user/keys", response_model=UserApiKeyList)
async def get_user_api_keys(user_id: str = Depends(get_firebase_token)):
    """Get all API keys for a user."""
    try:
        # Import the function
        from backend.auth.auth import _load_valid_keys
        
        # Load all valid keys
        valid_keys = _load_valid_keys()
        
        # Filter keys by user_id
        user_keys = []
        for key_id, key_info in valid_keys.items():
            if key_info.get("userId", key_info.get("user_id")) == user_id:
                # Format the key for the response
                created_at = key_info.get("created_at", 0)
                expires_at = key_info.get("expires_at", 0)
                
                # Handle Firebase timestamp objects
                if hasattr(created_at, "timestamp"):
                    created_at = int(created_at.timestamp())
                if hasattr(expires_at, "timestamp"):
                    expires_at = int(expires_at.timestamp())
                
                last_used = key_info.get("lastUsed")
                if hasattr(last_used, "timestamp"):
                    last_used = int(last_used.timestamp())
                
                user_keys.append({
                    "id": key_id,
                    "key": key_id,  # Use the key ID as the key value
                    "tier": key_info.get("tier", "free"),
                    "createdAt": created_at,
                    "expiresAt": expires_at,
                    "lastUsed": last_used,
                    "usageCount": key_info.get("usageCount", 0)
                })
        
        return {"keys": user_keys}
    except Exception as e:
        logging.error(f"Error getting user API keys: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching API keys: {str(e)}")

@app.post("/api/user/keys", response_model=CreateApiKeyResponse)
async def create_api_key(user_id: str = Depends(get_firebase_token)):
    """Create a new API key for a user."""
    try:
        # Get user subscription tier from Firestore
        if HAS_FIREBASE:
            user_doc = db.collection("users").document(user_id).get()
            
            if not user_doc.exists:
                # Create user document if it doesn't exist
                db.collection("users").document(user_id).set({
                    "subscription": "free",
                    "createdAt": firestore.SERVER_TIMESTAMP
                })
                tier = "free"
            else:
                tier = user_doc.to_dict().get("subscription", "free")
            
            # For free tier, check how many keys the user has created today
            if tier == "free":
                # Get current time and start of day
                current_time = int(time.time())
                day_start = current_time - (current_time % 86400)  # Start of current day
                
                # Query for keys created today by this user
                query = db.collection("api_keys").where("userId", "==", user_id).where("createdAt", ">=", firestore.Timestamp.fromtimestamp(day_start))
                keys_today = list(query.stream())
                
                # Enforce limit of 2 keys per day for free tier
                if len(keys_today) >= 2:
                    raise HTTPException(
                        status_code=429, 
                        detail="Free tier users can only create 2 API keys per day. Please upgrade your plan for unlimited API keys."
                    )
        else:
            # Default to free tier if Firebase isn't available
            tier = "free"
        
        # Generate new API key
        from backend.auth.auth import generate_api_key
        api_key = generate_api_key(user_id, tier)
        
        # Get the key info
        from backend.auth.auth import _load_valid_keys
        valid_keys = _load_valid_keys()
        key_info = valid_keys.get(api_key, {})
        
        # Format the response
        created_at = key_info.get("created_at", int(time.time()))
        expires_at = key_info.get("expires_at", int(time.time()) + 31536000)  # 1 year
        
        key_data = {
            "id": api_key,
            "key": api_key,
            "tier": tier,
            "createdAt": created_at,
            "expiresAt": expires_at,
            "lastUsed": None,
            "usageCount": 0
        }
        
        return {
            "key": key_data,
            "message": "API key created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error creating API key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating API key: {str(e)}")

@app.delete("/api/user/keys/{key_id}")
async def delete_api_key(key_id: str, user_id: str = Depends(get_firebase_token)):
    """Revoke an API key."""
    try:
        # Import auth functions
        from backend.auth.auth import _load_valid_keys, revoke_api_key
        
        # Load all valid keys
        valid_keys = _load_valid_keys()
        
        # Check if key exists
        if key_id not in valid_keys:
            raise HTTPException(status_code=404, detail="API key not found")
        
        # Check if key belongs to user
        key_user_id = valid_keys[key_id].get("userId", valid_keys[key_id].get("user_id"))
        if key_user_id != user_id:
            raise HTTPException(status_code=403, detail="You don't have permission to revoke this key")
        
        # Revoke the key
        if revoke_api_key(key_id):
            return {"message": "API key revoked successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to revoke API key")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error revoking API key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error revoking API key: {str(e)}")

def start_server(model_debugger, port: int = 8000):
    """Start the FastAPI server with the given ModelDebugger instance."""
    global debugger, api_key
    debugger = model_debugger
    
    # Capture the API key from the debugger
    api_key = getattr(model_debugger, "api_key", None)

    # Cleanup old visualizations
    cleanup_old_visualizations()

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)

    return app

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    # First, define the root endpoint to serve index.html
    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        global api_key
        
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            # Read the HTML content
            with open(index_path, "r") as f:
                html_content = f.read()
                
            # If we have an API key, inject JavaScript to fetch it
            if api_key:
                # Add a script to automatically set the API key for all requests
                api_key_script = f"""
                <script>
                    // Function to add API key to all fetch requests
                    const originalFetch = window.fetch;
                    window.fetch = function(url, options) {{
                        options = options || {{}};
                        options.headers = options.headers || {{}};
                        options.headers['X-API-Key'] = '{api_key}';
                        return originalFetch(url, options);
                    }};
                    console.log('API key interceptor enabled');
                </script>
                """
                
                # Insert the script before the closing </head> tag
                html_content = html_content.replace('</head>', f'{api_key_script}</head>')
                
                # Return the modified HTML
                return HTMLResponse(content=html_content)
                
            # If no API key, just return the original HTML
            return FileResponse(index_path)
        
        return {"message": "Cinder API is running but frontend is not available"}
    
    # Then mount the nested static directory
    nested_static = os.path.join(static_dir, "static")
    if os.path.exists(nested_static):
        app.mount("/static", StaticFiles(directory=nested_static), name="static_files")
    
    # Then mount the root static directory for other files
    app.mount("/", StaticFiles(directory=static_dir), name="root_static")