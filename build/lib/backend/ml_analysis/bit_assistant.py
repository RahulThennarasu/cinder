# Improved BitAssistant with more robust response handling

import os
import json
import re
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from fastapi import HTTPException

# Import Google's Generative AI library correctly
from google import genai

# Set up logger
logger = logging.getLogger(__name__)

class BitOptimizer:
    """Bit AI assistant for ML model analysis and optimization using Gemini API."""
    
    def __init__(self):
        """Initialize the BitAssistant with Gemini API."""
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = None
        
        # Log API key status
        if not self.api_key:
            logger.warning("No Gemini API key found. BitAssistant will be limited.")
        else:
            logger.info("Gemini API key found, initializing client")
            
        # Initialize the client
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize the Gemini API client."""
        if not self.api_key:
            return
            
        try:
            # Initialize the Google client
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Successfully initialized Gemini client for BitAssistant")
        except Exception as e:
            error_msg = f"Failed to initialize Gemini API client: {str(e)}"
            logger.error(error_msg)
            self.client = None
    
    async def analyze_model(self, model_code: str, framework: str) -> List[Dict[str, Any]]:
        """Analyze model code and identify optimization opportunities."""
        if not self.client:
            raise HTTPException(status_code=500, detail="Gemini API client not configured")
            
        # Create prompt for analysis - make it more structured
        prompt = f"""
        Analyze this {framework} machine learning model code and identify 3-5 specific optimization opportunities.
        
        ```python
        {model_code}
        ```
        
        For each optimization, provide:
        1. A concise title (e.g. "Add Dropout Regularization")
        2. A detailed explanation of why this change would improve the model
        3. Which specific part of the code should be modified
        4. The expected benefit of making this change
        
        Focus on common improvements like:
        - Regularization techniques (dropout, batch norm, etc.)
        - Architecture improvements (layer depth, width, connections)
        - Optimization enhancements (learning rate schedulers, better optimizers)
        - Training process improvements (early stopping, checkpointing)
        - Model efficiency optimizations
        
        You MUST format your response as JSON with the exact structure below:
        
        {{
          "optimizations": [
            {{
              "title": "Short title of the optimization",
              "description": "Detailed explanation of why this change is beneficial",
              "code_section": "Which part of the code to modify",
              "expected_benefit": "Specific improvements this change will provide"
            }},
            // additional optimizations follow the same format
          ]
        }}
        
        The JSON must be valid and parseable. Do not include explanations or markdown outside the JSON.
        """
        
        try:
            # Call Gemini API with retry logic
            optimizations = await self._call_gemini_with_retries(
                prompt, 
                parse_json=True,
                key="optimizations"
            )
            
            if not optimizations:
                # If parsing failed, create a fallback optimization
                logger.warning("Failed to parse optimizations from response, using fallback")
                optimizations = [{
                    "title": "Add Batch Normalization",
                    "description": "Batch normalization helps stabilize training by normalizing layer inputs, which allows for higher learning rates and faster convergence.",
                    "code_section": "Model definition class",
                    "expected_benefit": "Faster training, better generalization, and reduced sensitivity to initialization."
                }]
            
            return optimizations
            
        except HTTPException:
            # Pass through HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error analyzing model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error analyzing model: {str(e)}")
    
    async def generate_optimization_step(self, model_code: str, optimization: Dict[str, Any], framework: str) -> Dict[str, Any]:
        """Generate code changes for a specific optimization."""
        if not self.client:
            raise HTTPException(status_code=500, detail="Gemini API client not configured")
            
        # Create prompt for code modification - more strict formatting guidance
        prompt = f"""
        I need to apply the following optimization to this {framework} machine learning model:
        
        Optimization: {optimization['title']}
        Description: {optimization['description']}
        Code section to modify: {optimization['code_section']}
        
        Here is the current model code:
        
        ```python
        {model_code}
        ```
        
        Generate a detailed plan for implementing this optimization:
        1. Explain the specific code changes needed
        2. Provide the full updated code with the optimization applied
        3. Highlight exactly what was changed and why
        
        You MUST format your response as a valid JSON object with the EXACT structure shown below:
        
        {{
          "explanation": "Detailed explanation of the implementation approach",
          "updated_code": "The complete updated code with optimization applied",
          "changes_summary": "Summary of what was modified"
        }}
        
        The JSON must be properly formatted and parseable. The "updated_code" field must contain the complete functioning code with the optimization applied.
        Do not include additional explanations, markdown formatting, or any content outside the JSON structure.
        """
        
        try:
            # Call Gemini API with retry logic and more robust parsing
            result = await self._call_gemini_with_retries(
                prompt,
                parse_json=True
            )
            
            # If parsing failed or missing required fields, try to create a structured response
            if not result or "updated_code" not in result:
                logger.warning("Failed to parse code changes from response, attempting to extract code directly")
                
                # Try again with a simpler prompt
                simple_prompt = f"""
                Apply this optimization to the code: {optimization['title']}
                
                The original code is:
                ```python
                {model_code}
                ```
                
                Generate ONLY the complete updated code with the optimization applied.
                """
                
                # Get raw text response
                raw_response = await self._call_gemini_with_retries(
                    simple_prompt,
                    parse_json=False
                )
                
                # Extract code blocks from the response
                code_blocks = re.findall(r'```python\s*([\s\S]*?)\s*```', raw_response)
                if code_blocks:
                    updated_code = code_blocks[0].strip()
                else:
                    # Try to find any code-like content
                    updated_code = self._extract_code_without_markers(raw_response)
                
                # Create a fallback result
                if updated_code:
                    result = {
                        "explanation": f"Applied {optimization['title']} to improve model performance.",
                        "updated_code": updated_code,
                        "changes_summary": "Modified code to implement the optimization."
                    }
                else:
                    # If all fails, raise exception
                    raise Exception("Failed to extract code changes from response")
            
            # Clean up the code
            if "updated_code" in result:
                result["updated_code"] = self._clean_code(result["updated_code"])
            
            return result
            
        except HTTPException:
            # Pass through HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error generating optimization: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating optimization: {str(e)}")
    
    async def explain_optimization_benefits(self, original_code: str, updated_code: str, optimization: Dict[str, Any], framework: str) -> str:
        """Generate a detailed explanation of the benefits of an optimization."""
        if not self.client:
            raise HTTPException(status_code=500, detail="Gemini API client not configured")
            
        # Create prompt for explaining benefits
        prompt = f"""
        I've applied the following optimization to a {framework} machine learning model:
        
        Optimization: {optimization['title']}
        
        Original code section:
        ```python
        {original_code[:500]}  # Limit to avoid token limits
        ```
        
        Updated code section:
        ```python
        {updated_code[:500]}  # Limit to avoid token limits
        ```
        
        Explain in detail:
        1. What specific changes were made to implement this optimization
        2. How these changes improve the model's performance
        3. Any potential trade-offs or considerations to be aware of
        4. When this optimization is most effective
        
        Provide a comprehensive, technically accurate explanation for an ML engineer.
        """
        
        try:
            # Call Gemini API for explanation - text format is fine here
            explanation = await self._call_gemini_with_retries(
                prompt,
                parse_json=False
            )
            
            return explanation or f"Applied {optimization['title']} to improve model performance."
            
        except Exception as e:
            logger.error(f"Error explaining optimization: {str(e)}")
            return f"Error generating explanation: {str(e)}"
    
    async def _call_gemini_with_retries(self, 
                                       prompt: str, 
                                       max_retries: int = 3,
                                       parse_json: bool = False,
                                       key: str = None) -> Union[str, Dict[str, Any], List, None]:
        """Call Gemini API with retry logic and handle different response formats."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Calling Gemini API - attempt {attempt+1}/{max_retries}")
                
                # Call the API using the client
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash-preview-05-20",  # Using the same model as in SimpleCodeGenerator
                    contents=prompt,
                )
                
                # Extract text from response
                if hasattr(response, 'text'):
                    text_response = response.text
                elif hasattr(response, 'parts') and response.parts:
                    text_response = response.parts[0].text
                else:
                    logger.error(f"Unexpected response format: {type(response)}")
                    logger.error(f"Response attributes: {dir(response)}")
                    raise Exception("Unexpected response format from Gemini API")
                
                # Log a preview of the response for debugging
                logger.info(f"Response preview: {text_response[:200]}...")
                
                # Parse JSON if requested
                if parse_json:
                    result = self._extract_json_from_response(text_response, key)
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
                        wait_time = (2 ** attempt) + 0.5
                        logger.warning(f"Rate limited. Retrying in {wait_time:.1f} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # All retries failed
                        logger.error("Rate limit retries exhausted")
                        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
                else:
                    # For other errors, retry with a more direct approach if possible
                    if attempt < max_retries - 1:
                        logger.warning(f"Error occurred, retrying with simplified prompt")
                        # Simplify the prompt
                        if parse_json:
                            # Add stronger formatting instructions
                            prompt = self._simplify_prompt_for_retry(prompt)
                        await asyncio.sleep(1)
                        continue
                    else:
                        # All retries failed
                        raise
        
        # If we reach here, all retries failed
        logger.error("All API call attempts failed")
        return None
    
    def _simplify_prompt_for_retry(self, original_prompt: str) -> str:
        """Simplify a prompt for retry after failure."""
        # Remove complex instructions, focus on core request
        lines = original_prompt.strip().split('\n')
        simplified = []
        
        # Keep the first few lines that describe the task
        for i, line in enumerate(lines[:5]):
            if line.strip():
                simplified.append(line)
        
        # Add more direct JSON formatting instructions
        simplified.append("\nYou MUST return your response as a valid JSON object.")
        simplified.append("The JSON should be properly formatted with double quotes around keys and string values.")
        simplified.append("DO NOT include markdown code blocks, just the raw JSON object.")
        
        return "\n".join(simplified)
    
    def _extract_json_from_response(self, text: str, key: str = None) -> Union[Dict[str, Any], List, None]:
        """Extract and parse JSON from Gemini response text with multiple strategies."""
        try:
            # Strategy 1: Find JSON in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            
            if json_match:
                json_str = json_match.group(1)
                try:
                    parsed = json.loads(json_str)
                    logger.info("Successfully parsed JSON from markdown code block")
                    if key and key in parsed:
                        return parsed[key]
                    return parsed
                except json.JSONDecodeError:
                    logger.warning("Found markdown code block but couldn't parse JSON")
            
            # Strategy 2: Find JSON without markdown - match from first { to last }
            try:
                first_brace = text.find('{')
                last_brace = text.rfind('}')
                if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                    json_str = text[first_brace:last_brace+1]
                    parsed = json.loads(json_str)
                    logger.info("Successfully parsed JSON without markdown")
                    if key and key in parsed:
                        return parsed[key]
                    return parsed
            except json.JSONDecodeError:
                logger.warning("Found braces but couldn't parse complete JSON")
            
            # Strategy 3: Try to find and fix common JSON formatting issues
            try:
                # Replace single quotes with double quotes (common issue)
                fixed_text = re.sub(r'(\w+)\':', r'"\1":', text)
                fixed_text = re.sub(r': \'([^\']+)\'', r': "\1"', fixed_text)
                
                # Find JSON-like structure
                fixed_match = re.search(r'{[\s\S]*?}', fixed_text)
                if fixed_match:
                    json_str = fixed_match.group(0)
                    parsed = json.loads(json_str)
                    logger.info("Successfully parsed JSON after fixing formatting")
                    if key and key in parsed:
                        return parsed[key]
                    return parsed
            except json.JSONDecodeError:
                logger.warning("Attempted to fix JSON but still couldn't parse")
            
            # Strategy 4: For array responses specifically
            if key:
                array_match = re.search(r'\[\s*{[\s\S]*?}\s*\]', text)
                if array_match:
                    try:
                        json_str = array_match.group(0)
                        parsed = json.loads(json_str)
                        logger.info("Successfully parsed JSON array directly")
                        return parsed
                    except json.JSONDecodeError:
                        logger.warning("Found array but couldn't parse JSON")
            
            logger.warning(f"Failed to parse JSON with all strategies: {text[:100]}...")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error extracting JSON: {str(e)}")
            return None
    
    def _clean_code(self, code: str) -> str:
        """Clean up code from markdown formatting."""
        # Remove markdown code blocks
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        return code
    
    def _extract_code_without_markers(self, text: str) -> str:
        """Extract Python code from text when markdown markers are missing."""
        # Try to find Python-like content (imports, function definitions)
        python_patterns = [
            r'import\s+[\w\s,]+', 
            r'def\s+\w+\s*\([\w\s,=\[\]\'\"]*\):',
            r'class\s+\w+\s*[\(\w\s,=\[\]\'\"]*:',
            r'if\s+__name__\s*==\s*[\'"]__main__[\'"]:',
            r'torch\.nn\.Module',
            r'tf\.keras\.Model'
        ]
        
        # Find the earliest match of any pattern
        earliest_match = None
        earliest_pos = float('inf')
        
        for pattern in python_patterns:
            match = re.search(pattern, text)
            if match and match.start() < earliest_pos:
                earliest_match = match
                earliest_pos = match.start()
        
        if earliest_match:
            # Extract from this point to the end
            return text[earliest_pos:].strip()
        
        # If no patterns match, return empty string
        return ""