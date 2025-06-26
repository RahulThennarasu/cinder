#!/usr/bin/env python3
"""
CLI for Cinder - ML Model Analysis Dashboard
"""

import argparse
import os
import sys
import importlib.util
import logging
import time

# Add or update this section in cinder/cli.py

def generate_key(args):
    """Generate a new API key."""
    try:
        from backend.auth.auth import generate_api_key, list_valid_keys
        
        user_id = args.user_id or f"user_{int(time.time())}"
        
        # Generate the key (this now automatically adds it to the valid keys list)
        api_key = generate_api_key(user_id)
        
        print(f"Generated API key for {user_id}:")
        print(f"API Key: {api_key}")
        print("\nKey has been automatically added to valid keys.")
        print("Keep this key secure and do not share it.")
        print("You can set it as an environment variable:")
        print(f"export CINDER_API_KEY={api_key}")
        
        # Offer to save to .env file
        if args.save_to_env:
            env_path = os.path.join(os.getcwd(), ".env")
            try:
                # Check if .env exists
                if os.path.exists(env_path):
                    with open(env_path, "r") as f:
                        env_content = f.read()
                    
                    # Check if CINDER_API_KEY already exists
                    if "CINDER_API_KEY=" in env_content:
                        print("\nWarning: CINDER_API_KEY already exists in .env file.")
                        response = input("Do you want to overwrite it? (y/n): ").lower()
                        if response != 'y':
                            print("API key not saved to .env file.")
                            return
                        
                        # Replace existing key
                        env_content = env_content.replace(
                            env_content.split("CINDER_API_KEY=")[1].split("\n")[0],
                            api_key
                        )
                        with open(env_path, "w") as f:
                            f.write(env_content)
                    else:
                        # Add new key
                        with open(env_path, "a") as f:
                            f.write(f"\nCINDER_API_KEY={api_key}\n")
                else:
                    # Create new .env file
                    with open(env_path, "w") as f:
                        f.write(f"CINDER_API_KEY={api_key}\n")
                
                print(f"\nAPI key saved to {env_path}")
            except Exception as e:
                print(f"Error saving to .env file: {e}")
        
    except ImportError:
        print("Error: Authentication module not available.")
        print("Make sure you have installed the 'backend.auth' module.")

def list_keys(args):
    """List all valid API keys."""
    try:
        from backend.auth.auth import list_valid_keys
        
        keys = list_valid_keys()
        
        if not keys:
            print("No valid API keys found.")
            return
        
        print("\nValid API Keys:")
        print("=" * 80)
        print(f"{'Masked Key':<30} {'User ID':<20} {'Created':<20} {'Expires':<20}")
        print("-" * 80)
        
        for key_info in keys:
            created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(key_info["created_at"]))
            expires = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(key_info["expires_at"]))
            
            print(f"{key_info['key']:<30} {key_info['user_id']:<20} {created:<20} {expires:<20}")
        
    except ImportError:
        print("Error: Authentication module not available.")
        print("Make sure you have installed the 'backend.auth' module.")

def revoke_key(args):
    """Revoke an API key."""
    try:
        from backend.auth.auth import revoke_api_key, list_valid_keys
        
        # If no key is provided, list keys and ask which one to revoke
        if not args.api_key:
            keys = list_valid_keys()
            
            if not keys:
                print("No valid API keys found.")
                return
            
            print("\nValid API Keys:")
            for i, key_info in enumerate(keys, 1):
                print(f"{i}. {key_info['key']} ({key_info['user_id']})")
            
            try:
                choice = int(input("\nEnter the number of the key to revoke (or 0 to cancel): "))
                if choice == 0:
                    print("Operation cancelled.")
                    return
                
                if choice < 1 or choice > len(keys):
                    print("Invalid choice.")
                    return
                
                # Get the actual key (this is just a demonstration - in reality you'd need to store the full keys)
                print("In a real implementation, you'd need to store the full keys.")
                print("This demo can't actually revoke keys since we only have masked versions.")
                return
                
            except ValueError:
                print("Invalid input.")
                return
        
        # Revoke the specified key
        if revoke_api_key(args.api_key):
            print(f"API key {args.api_key[:5]}...{args.api_key[-3:]} has been revoked.")
        else:
            print(f"Failed to revoke API key. Key may not exist.")
        
    except ImportError:
        print("Error: Authentication module not available.")
        print("Make sure you have installed the 'backend.auth' module.")

def main():
    """Main entry point for the Cinder CLI."""
    parser = argparse.ArgumentParser(description="Cinder - ML Model Analysis Dashboard")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Existing commands...
    run_parser = subparsers.add_parser("run", help="Run an example script")
    run_parser.add_argument("example", choices=["pytorch", "sklearn", "tensorflow", "quickstart"], 
                        help="Example script to run")
    
    serve_parser = subparsers.add_parser("serve", help="Start the dashboard server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    
    # API key management commands
    key_parser = subparsers.add_parser("generate-key", help="Generate a new API key")
    key_parser.add_argument("--user-id", help="User identifier for the key")
    key_parser.add_argument("--save-to-env", action="store_true", help="Save the key to .env file")
    
    list_parser = subparsers.add_parser("list-keys", help="List all valid API keys")
    
    revoke_parser = subparsers.add_parser("revoke-key", help="Revoke an API key")
    revoke_parser.add_argument("--api-key", help="The API key to revoke")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "run":
        run_example(args.example)
    elif args.command == "serve":
        start_server(args.host, args.port)
    elif args.command == "generate-key":
        generate_key(args)
    elif args.command == "list-keys":
        list_keys(args)
    elif args.command == "revoke-key":
        revoke_key(args)
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