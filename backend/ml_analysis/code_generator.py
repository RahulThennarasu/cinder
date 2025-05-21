import os
import requests
import json
from typing import Dict, Any, Optional

class SimpleCodeGenerator:
    """Simple code example generator using templates with optional Gemini API fallback."""
    
    def __init__(self, api_key=None):
        """Initialize with Gemini API key."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            print("Warning: No Gemini API key provided. Using predefined templates only.")
    
    def generate_code_example(self, 
                             framework: str, 
                             category: str, 
                             model_context: Dict[str, Any]) -> str:
        """Generate code example for ML model improvement."""
        # First check if we have a template
        template_code = self._get_template_code(framework, category)
        if template_code:
            print(f"Using template code for {framework} - {category}")
            return template_code
            
        # If no template and no API key, return fallback
        if not self.api_key:
            return f"# Code example generation unavailable - API key not configured"
            
        # Otherwise try the API
        try:
            # Prepare the prompt with context
            accuracy = model_context.get('accuracy', 0)
            error_rate = model_context.get('error_rate', 0)
            framework_name = {
                'pytorch': 'PyTorch',
                'tensorflow': 'TensorFlow',
                'sklearn': 'scikit-learn'
            }.get(framework, framework)
            
            prompt = f"""
            You are an expert ML developer. Generate a clean code example in {framework_name} for {category}.
            
            Model context:
            - Accuracy: {accuracy:.4f}
            - Error rate: {error_rate:.4f}
            
            Return ONLY the code example without explanations before or after.
            """
            
            # Direct API call to Gemini
            url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 2048,
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            # Check for rate limiting
            if response.status_code == 429:
                print("Rate limit exceeded. Using fallback code.")
                return self._get_fallback_code(framework, category)
            
            if response.status_code == 200:
                result = response.json()
                # Extract text from the response
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']
                    if 'parts' in content and len(content['parts']) > 0:
                        code = content['parts'][0]['text']
                        # Clean up the response
                        code = code.replace("```python", "").replace("```", "").strip()
                        return code
            
            # If we get here, something went wrong with the API call
            print(f"API error: {response.status_code}")
            return self._get_fallback_code(framework, category)
            
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return self._get_fallback_code(framework, category)
    
    def _get_template_code(self, framework, category):
        """Get predefined template code for common frameworks and categories."""
        # Simplified templates just for demonstration
        templates = {
            'pytorch': {
                'use_cross_validation': """# PyTorch cross-validation example
import torch
import numpy as np
from sklearn.model_selection import KFold

def cross_validate(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        # Train model on each fold
        # Evaluate and collect scores
        pass
        
    return scores"""
            },
            'tensorflow': {
                'use_cross_validation': """# TensorFlow cross-validation example
import tensorflow as tf
from sklearn.model_selection import KFold

def cross_validate(model_fn, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        # Create and train model on each fold
        # Evaluate and collect scores
        pass
        
    return scores"""
            },
            'sklearn': {
                'use_cross_validation': """# scikit-learn cross-validation example
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f}")"""
            }
        }
        
        if framework in templates and category in templates[framework]:
            return templates[framework][category]
        
        return None
    
    def _get_fallback_code(self, framework, category):
        """Get generic fallback code when rate limited or no template."""
        fallback_templates = {
            'pytorch': f"""# PyTorch placeholder for {category}
import torch
import torch.nn as nn

# Basic implementation example would go here
# Check PyTorch documentation for specific details""",

            'tensorflow': f"""# TensorFlow placeholder for {category}
import tensorflow as tf

# Basic implementation example would go here
# Check TensorFlow documentation for specific details""",

            'sklearn': f"""# scikit-learn placeholder for {category}
# Basic implementation example would go here
# Check scikit-learn documentation for specific details"""
        }
        
        return fallback_templates.get(framework, f"# Code example not available")