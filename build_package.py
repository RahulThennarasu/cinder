#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path

def main():
    # Get the root directory of the project
    root_dir = Path(__file__).parent.absolute()
    
    # Clean up previous build artifacts
    build_dir = root_dir / "build"
    dist_dir = root_dir / "dist"
    static_dir = root_dir / "cinder" / "app" / "static"
    
    for dir_path in [build_dir, dist_dir]:
        if dir_path.exists():
            print(f"Cleaning {dir_path}")
            shutil.rmtree(dir_path)
    
    # Create the static directory if it doesn't exist
    static_dir.mkdir(parents=True, exist_ok=True)
    
    # Build the React frontend
    frontend_dir = root_dir / "frontend"
    if frontend_dir.exists():
        print("Building React frontend...")
        os.chdir(frontend_dir)
        subprocess.run(["npm", "install"], check=True)
        subprocess.run(["npm", "run", "build"], check=True)
        
        # Copy the build to the static directory
        build_output = frontend_dir / "build"
        if build_output.exists():
            print(f"Copying frontend build to {static_dir}")
            for item in build_output.glob("*"):
                if item.is_dir():
                    if (static_dir / item.name).exists():
                        shutil.rmtree(static_dir / item.name)
                    shutil.copytree(item, static_dir / item.name)
                else:
                    shutil.copy2(item, static_dir)
        else:
            print("Frontend build output not found. Skipping...")
    else:
        print("Frontend directory not found. Skipping frontend build...")
    
    # Check if build package is installed
    try:
        import build
    except ImportError:
        print("Installing build package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "build", "hatch"], check=True)
    
    # Build the Python package
    os.chdir(root_dir)
    print("Building Python package...")
    subprocess.run([sys.executable, "-m", "build"], check=True)
    
    print("\nBuild completed successfully!")
    print(f"Package files available in {dist_dir}")

if __name__ == "__main__":
    main()