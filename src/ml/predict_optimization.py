#!/usr/bin/env python3
"""
Prediction script for the ML optimization model.
This script is called by the LLVM pass to get optimization decisions.
"""

import sys
import json
import numpy as np
import os
from bandit_model import LinUCBOptimizer, extract_feature_vector

def main():
    """
    Main function to predict the optimization level for a function
    
    Usage: predict_optimization.py <function_name>
    
    The function features should be in a file named 'features.json'
    """
    if len(sys.argv) < 2:
        print("Usage: predict_optimization.py <function_name>")
        sys.exit(1)
    
    function_name = sys.argv[1]
    
    # Default to -O3 if anything goes wrong
    default_opt_level = 3
    
    # Path to the model
    model_path = os.path.join(
        os.environ.get('COMPILEML_PATH', '.'),
        'data',
        'model.pkl'
    )
    
    # Check if model exists
    if not os.path.exists(model_path):
        # No model found, return default
        print(default_opt_level)
        sys.exit(0)
    
    # Try to load features
    try:
        with open('features.json', 'r') as f:
            features_data = json.load(f)
    except (IOError, json.JSONDecodeError):
        # Failed to load features, return default
        print(default_opt_level)
        sys.exit(0)
    
    # Check if the function is in the features data
    if function_name not in features_data:
        # Function not found, return default
        print(default_opt_level)
        sys.exit(0)
    
    # Extract features for the function
    function_features = features_data[function_name]
    feature_vector = extract_feature_vector(function_features)
    
    try:
        # Load model
        model = LinUCBOptimizer.load(model_path)
        
        # Get prediction
        opt_level = model.select_action(feature_vector)
        
        # Print the result to stdout
        print(opt_level)
    except Exception:
        # On any error, return default
        print(default_opt_level)


if __name__ == "__main__":
    main()
