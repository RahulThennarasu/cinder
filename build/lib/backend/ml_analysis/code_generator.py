import os
import json
import time
import logging
from typing import Dict, Any, Optional

from google import genai  # This is the updated import

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CodeGenerator")

class SimpleCodeGenerator:
    """Code example generator using Gemini API with basic fallbacks."""
    
    def __init__(self, api_key=None):
        """Initialize with Gemini API key."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("No Gemini API key provided. Code generation will not be available.")
        
        # Initialize the Google client
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Successfully initialized Gemini client")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {str(e)}")
                self.client = None
        else:
            self.client = None
        
        # Add a flag to track rate limiting
        self.rate_limited = False
        
        # Add a simple in-memory cache
        self.code_cache = {}
    
    def generate_code_example(self, 
                             framework: str, 
                             category: str, 
                             model_context: Dict[str, Any] = None) -> str:
        """Generate code example for ML model improvement using Gemini API with fallbacks."""
        # Default empty dict if model_context is None
        model_context = model_context or {}
        
        logger.info(f"Generating code example for framework={framework}, category={category}")
        
        # Standardize keys for better cache hits
        framework = framework.lower().strip()
        category = category.lower().strip().replace(" ", "_")
        
        # Create a cache key
        cache_key = f"{framework}_{category}"
        
        # Check if we have this in cache
        if cache_key in self.code_cache:
            logger.info(f"Found cached result for {cache_key}")
            return self.code_cache[cache_key]
        
        # If client is not initialized, return error message
        if not self.client:
            logger.error("Cannot generate code - Gemini client not initialized")
            return "# Code example generation unavailable - API client not configured"
            
        try:
            # Prepare the prompt with detailed context
            accuracy = model_context.get('accuracy', 0)
            error_rate = model_context.get('error_rate', 0)
            
            framework_name = {
                'pytorch': 'PyTorch',
                'tensorflow': 'TensorFlow',
                'sklearn': 'scikit-learn'
            }.get(framework, framework)
            
            category_display = category.replace('_', ' ').title()
            
            # Create a more detailed prompt for better results
            prompt = f"""
            As an expert ML developer, write a complete, production-ready {framework_name} implementation for {category_display}.
            
            Model details:
            - Current accuracy: {accuracy:.4f}
            - Error rate: {error_rate:.4f}
            - Framework: {framework_name}
            
            The code should:
            - Be well-organized and follow best practices
            - Include proper comments and docstrings
            - Be ready to run with minimal modifications
            - Handle edge cases appropriately
            
            Return ONLY the executable code without additional explanations before or after.
            """
            
            # Use the client to generate content - with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Calling Gemini API - attempt {attempt+1}/{max_retries}")
                    
                    # Call the Gemini API using the client
                    response = self.client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=prompt,
                    )
                    
                    # Log the response type for debugging
                    logger.info(f"Received response of type: {type(response)}")
                    logger.info(f"Response has text attribute: {hasattr(response, 'text')}")
                    
                    # Extract the generated code
                    if hasattr(response, 'text'):
                        code = response.text
                        
                        # Clean up the code (remove markdown code block markers if present)
                        code = code.replace("```python", "").replace("```", "").strip()
                        
                        # Cache the generated code
                        self.code_cache[cache_key] = code
                        
                        logger.info(f"Successfully generated code for {cache_key}")
                        return code
                    elif hasattr(response, 'parts') and response.parts:
                        # For newer Gemini API versions, text might be in parts
                        code = response.parts[0].text
                        
                        # Clean up the code (remove markdown code block markers if present)
                        code = code.replace("```python", "").replace("```", "").strip()
                        
                        # Cache the generated code
                        self.code_cache[cache_key] = code
                        
                        logger.info(f"Successfully generated code for {cache_key} using parts")
                        return code
                    else:
                        # Log the response for debugging
                        logger.error(f"Unexpected response format: {response}")
                        logger.error(f"Response dir: {dir(response)}")
                        
                        return "# Error: Unexpected response format from Gemini API"
                    
                except Exception as e:
                    error_str = str(e)
                    logger.error(f"Error on attempt {attempt+1}: {error_str}")
                    
                    # Check if it's a rate limit error
                    if "quota" in error_str.lower() or "rate" in error_str.lower() or "429" in error_str:
                        if attempt < max_retries - 1:
                            # Calculate backoff with exponential delay
                            wait_time = (2 ** attempt) + 0.5
                            logger.warning(f"Rate limited. Retrying in {wait_time:.1f} seconds...")
                            time.sleep(wait_time)
                            continue
                        else:
                            # All retries failed
                            self.rate_limited = True
                            logger.error("Rate limit retries exhausted")
                            return "# Rate limit exceeded. Please try again later."
                    else:
                        # Non-rate-limit error
                        return f"# Error: {error_str}"
            
            # If we get here, something unexpected happened
            logger.error("Failed to generate code after all retries")
            return "# Could not generate code. Please try again later."
                
        except Exception as e:
            error_message = f"Error generating code: {str(e)}"
            logger.exception("Unexpected exception in generate_code_example")
            return f"# {error_message}"