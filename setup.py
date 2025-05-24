#!/usr/bin/env python3

import os
import sys
from setuptools import setup, find_packages

# Read version from __init__.py
with open("backend/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break
    else:
        version = "0.1.0"

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Define requirements
requirements = [
    "fastapi>=0.95.0",
    "uvicorn>=0.15.0",
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.5.0",
    "python-dotenv>=1.0.0",
]

# Define extras
extras_require = {
    "pytorch": ["torch>=1.12.0"],
    "tensorflow": ["tensorflow>=2.8.0"],
    "all": [
        "torch>=1.12.0",
        "tensorflow>=2.8.0",
        "google-generativeai>=0.1.0"
    ],
}

# Get packages
packages = find_packages(exclude=["tests*", "examples*"])

# Setup
setup(
    name="ml-dashboard",
    version=version,
    author="Rahul Thennarasu",
    author_email="rahulthennarasu07@gmail.com",
    description="A machine learning model analysis dashboard with AI assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RahulThennarasu/ml-dashboard",
    packages=packages,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extras_require,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "ml-dashboard=backend.app.cli:main",
        ],
    },
)