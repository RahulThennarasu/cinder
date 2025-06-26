from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import json
import re
import logging
import asyncio
from google import genai  # Updated import for newer Gemini client

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BitRouter")

# Initialize the Gemini client
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    try:
        genai_client = genai.Client(api_key=api_key)
        logger.info("Successfully initialized Gemini client for Bit")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {str(e)}")
        genai_client = None
else:
    logger.warning("No Gemini API key provided. Bit chat will use fallback responses.")
    genai_client = None

# Create router
router = APIRouter()

# Define request and response models
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

@router.post("/api/bit-chat", response_model=BitChatResponse)
async def bit_chat(request: BitChatRequest = Body(...)):
    """Process a chat request from Bit and return AI-generated responses"""
    try:
        if not genai_client:
            return generate_fallback_response(request)
        
        # Format prompt for Gemini
        prompt = f"""
        You are Bit, an AI assistant specialized in analyzing and improving machine learning code.
        
        Current code to analyze:
        ```python
        {request.code}
        ```
        
        Model details:
        - Framework: {request.framework}
        - Accuracy: {request.modelInfo.get('accuracy', 'unknown') if request.modelInfo else 'unknown'}
        - Precision: {request.modelInfo.get('precision', 'unknown') if request.modelInfo else 'unknown'}
        - Recall: {request.modelInfo.get('recall', 'unknown') if request.modelInfo else 'unknown'}
        
        User query: {request.query}
        
        Provide a helpful response about the code and the user's query.
        If you have specific suggestions to improve the code, include them in the suggestions array.
        
        Format your response as JSON with these fields:
        {{
          "message": "Your main response text",
          "suggestions": [
            {{
              "title": "Suggestion title",
              "description": "Detailed explanation",
              "code": "Suggested code fix",
              "lineNumber": 10 // Line number to apply the fix
            }}
          ]
        }}
        
        Make sure your response is valid JSON and can be parsed. Remember, the code should be in plain text, not markdown.
        """
        
        # Call Gemini API with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Calling Gemini API - attempt {attempt+1}/{max_retries}")
                
                # Generate content using the client
                response = genai_client.models.generate_content(
                    model="gemini-1.5-flash",  # or gemini-2.5-flash-preview-05-20 if available
                    contents=prompt
                )
                
                # Extract the generated text
                if hasattr(response, 'text'):
                    text = response.text
                elif hasattr(response, 'parts') and response.parts:
                    text = response.parts[0].text
                else:
                    logger.error(f"Unexpected response format: {response}")
                    logger.error(f"Response dir: {dir(response)}")
                    return generate_fallback_response(request)
                
                # Try to extract and parse JSON from the response
                json_match = re.search(r'```json([\s\S]*?)```', str(text)) if text else None or re.search(r'{[\s\S]*}', str(text)) if text else None
                json_text = json_match.group(0).replace('```json', '').replace('```', '') if json_match else str(text)
                
                try:
                    parsed_response = json.loads(json_text)
                    
                    # Ensure the response has the required fields
                    if "message" not in parsed_response:
                        parsed_response["message"] = "I've analyzed your code."
                    
                    if "suggestions" not in parsed_response:
                        parsed_response["suggestions"] = []
                    
                    # Validate suggestions format
                    for suggestion in parsed_response["suggestions"]:
                        if "lineNumber" not in suggestion:
                            # Try to infer a reasonable line number if missing
                            code_lines = request.code.split("\n")
                            for i, line in enumerate(code_lines):
                                if any(keyword in line.lower() for keyword in ["def", "class", "import", "model"]):
                                    suggestion["lineNumber"] = i + 1
                                    break
                            if "lineNumber" not in suggestion:
                                suggestion["lineNumber"] = 1  # Default to line 1
                    
                    return parsed_response
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {e}, text: {json_text[:100]}...")
                    
                    if attempt < max_retries - 1:
                        continue  # Try again
                    
                    # If all retries fail to parse JSON, return a formatted response
                    return {
                        "message": str(text),
                        "suggestions": []
                    }
                
            except Exception as e:
                error_str = str(e)
                logger.error(f"Error on attempt {attempt+1}: {error_str}")
                
                # Check if it's a rate limit error
                if "quota" in error_str.lower() or "rate" in error_str.lower() or "429" in error_str:
                    if attempt < max_retries - 1:
                        # Calculate backoff with exponential delay
                        wait_time = (2 ** attempt) + 0.5
                        logger.warning(f"Rate limited. Retrying in {wait_time:.1f} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                
                if attempt < max_retries - 1:
                    continue  # Try again for other errors too
                
                # If all retries fail, return a fallback response
                return generate_fallback_response(request)
        
        # If we get here, all retries failed
        return generate_fallback_response(request)
    
    except Exception as e:
        logger.exception(f"Unexpected error in bit_chat: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}"
        )

def generate_fallback_response(request: BitChatRequest) -> Dict[str, Any]:
    """Generate a fallback response when Gemini API is unavailable"""
    query = request.query.lower()
    code = request.code
    framework = request.framework.lower()
    
    # Check for common keywords and provide appropriate responses
    if any(word in query for word in ["improve", "enhance", "better", "optimize"]):
        return {
            "message": "Based on my analysis, I can suggest a few improvements to your model code:",
            "suggestions": [
                {
                    "title": "Add Regularization",
                    "description": "Your model might benefit from regularization to prevent overfitting. Consider adding dropout layers.",
                    "code": "self.dropout = nn.Dropout(0.3)\n# Use in forward pass after activation" if framework == "pytorch" else 
                           "model.add(tf.keras.layers.Dropout(0.3))" if framework == "tensorflow" else
                           "model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced')",
                    "lineNumber": 7
                }
            ]
        }
    elif any(word in query for word in ["explain", "understand", "what", "how"]):
        return {
            "message": f"Your code implements a neural network model using {request.framework}. It defines a basic architecture with input, hidden, and output layers. The model is trained using standard optimization techniques. I notice it's a classification model with {request.modelInfo.get('accuracy', 0)*100:.1f}% accuracy.",
            "suggestions": []
        }
    elif any(word in query for word in ["performance", "accuracy", "precision", "metrics"]):
        return {
            "message": f"Your model currently has {request.modelInfo.get('accuracy', 0)*100:.1f}% accuracy. To improve performance, consider increasing model complexity, adding more training data, or fine-tuning hyperparameters.",
            "suggestions": [
                {
                    "title": "Increase Model Complexity",
                    "description": "Adding more layers can help the model learn more complex patterns.",
                    "code": "self.layer1 = nn.Linear(input_size, hidden_size)\nself.layer2 = nn.Linear(hidden_size, hidden_size//2)\nself.layer3 = nn.Linear(hidden_size//2, num_classes)" if framework == "pytorch" else
                            "model.add(tf.keras.layers.Dense(128, activation='relu'))\nmodel.add(tf.keras.layers.Dense(64, activation='relu'))" if framework == "tensorflow" else
                            "model = RandomForestClassifier(n_estimators=200, max_depth=15)",
                    "lineNumber": 6
                }
            ]
        }
    else:
        # Generic response
        return {
            "message": f"I've looked at your {request.framework} code. It's a machine learning model with standard architecture. How can I help you improve it?",
            "suggestions": []
        }