#!/usr/bin/env python3
"""
ML Dashboard - Command Line Interface

This module provides command-line utilities for the ML Dashboard.
"""

import os
import sys
import argparse
import importlib.util
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ml-dashboard-cli")

def load_model_from_file(file_path: str, model_var_name: str = "model", dataset_var_name: str = "dataset"):
    """
    Load a model and dataset from a Python file.
    
    Args:
        file_path: Path to the Python file containing the model and dataset
        model_var_name: Variable name for the model in the file
        dataset_var_name: Variable name for the dataset in the file
    
    Returns:
        tuple: (model, dataset) if found, otherwise (None, None)
    """
    try:
        # Make the file path absolute
        file_path = os.path.abspath(file_path)
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None, None
            
        # Extract the module name from the file path
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Extract the model and dataset
        model = getattr(module, model_var_name, None)
        dataset = getattr(module, dataset_var_name, None)
        
        if model is None:
            logger.error(f"Model variable '{model_var_name}' not found in {file_path}")
        
        if dataset is None:
            logger.error(f"Dataset variable '{dataset_var_name}' not found in {file_path}")
        
        return model, dataset
    
    except Exception as e:
        logger.error(f"Error loading model from file: {str(e)}")
        return None, None

def main():
    """Main entry point for the ML Dashboard CLI."""
    parser = argparse.ArgumentParser(description="ML Dashboard CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Launch command
    launch_parser = subparsers.add_parser("launch", help="Launch the dashboard for an existing model")
    launch_parser.add_argument("file", help="Python file containing the model and dataset")
    launch_parser.add_argument("--model", help="Variable name for the model", default="model")
    launch_parser.add_argument("--dataset", help="Variable name for the dataset", default="dataset")
    launch_parser.add_argument("--port", type=int, help="Port to run the dashboard on", default=8000)
    launch_parser.add_argument("--name", help="Name for the dashboard", default="My Model")
    launch_parser.add_argument("--api-key", help="Gemini API key for code generation", default=None)
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Display the dashboard version")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command is specified, display help
    if args.command is None:
        parser.print_help()
        return
    
    # Handle version command
    if args.command == "version":
        from backend import __version__
        print(f"ML Dashboard v{__version__}")
        return
    
    # Handle launch command
    if args.command == "launch":
        # Load the model and dataset
        model, dataset = load_model_from_file(args.file, args.model, args.dataset)
        
        if model is None or dataset is None:
            print("Error loading model or dataset. Please check your file and variable names.")
            return 1
        
        # Launch the dashboard
        try:
            from backend import ModelDashboard
            dashboard = ModelDashboard(
                model=model,
                dataset=dataset,
                name=args.name,
                api_key=args.api_key
            )
            dashboard.launch(port=args.port)
        except Exception as e:
            logger.error(f"Error launching dashboard: {str(e)}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())