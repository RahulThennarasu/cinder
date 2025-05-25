#!/usr/bin/env python3
"""
CLI for Cinder - ML Model Analysis Dashboard
"""

import argparse
import os
import sys
import importlib.util
import logging

def main():
    """Main entry point for the Cinder CLI."""
    parser = argparse.ArgumentParser(description="Cinder - ML Model Analysis Dashboard")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # 'run' command to run examples
    run_parser = subparsers.add_parser("run", help="Run an example script")
    run_parser.add_argument("example", choices=["pytorch", "sklearn", "tensorflow", "quickstart"], 
                        help="Example script to run")
    
    # 'serve' command to start the dashboard directly
    serve_parser = subparsers.add_parser("serve", help="Start the dashboard server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "run":
        run_example(args.example)
    elif args.command == "serve":
        start_server(args.host, args.port)
    else:
        # No command provided, show help
        parser.print_help()

def run_example(example_name):
    """Run one of the example scripts."""
    example_files = {
        "pytorch": "examples/high_variance.py",
        "sklearn": "examples/scikit_demo.py",
        "tensorflow": "examples/tensorflow_demo.py",
        "quickstart": "examples/run_server.py"
    }
    
    if example_name not in example_files:
        print(f"Error: Unknown example '{example_name}'")
        return
    
    file_path = example_files[example_name]
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: Example file not found: {file_path}")
        return
    
    print(f"Running example: {example_name}")
    
    # Execute the example file
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location("example_module", file_path)
        if spec is None:
            print(f"Error: Could not load module from {file_path}")
            return
            
        example_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            print(f"Error: Module loader is None for {file_path}")
            return
            
        spec.loader.exec_module(example_module)
        
        # Call the main function if it exists
        if hasattr(example_module, "main"):
            example_module.main()
        else:
            print("Warning: No 'main' function found in the example file.")
    except Exception as e:
        print(f"Error running example: {e}")

def start_server(host, port):
    """Start the Cinder dashboard server directly."""
    try:
        import uvicorn
        from backend.app.server import app
        
        print(f"Starting Cinder dashboard server at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        print("Error: Required packages not found. Please install uvicorn and fastapi.")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()