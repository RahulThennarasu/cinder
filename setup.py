from setuptools import setup, find_packages
import os

# Create a list of all packages to include
packages = ['cinder']

# Add backend packages manually
backend_packages = [
    'backend',
    'backend.app',
    'backend.ml_analysis',
    'backend.model_interface',
    'backend.auth',
]

# Add backend packages only if they exist
for pkg in backend_packages:
    pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), pkg.replace('.', os.path.sep))
    if os.path.isdir(pkg_dir):
        packages.append(pkg)
    else:
        print(f"Warning: Package directory not found: {pkg_dir}")

setup(
    name="cinder-ml",
    version="1.8.5",  # Start with version 1.0.0
    description="ML model debugging and analysis dashboard",
    author="Rahul Thennarasu",
    author_email="rahulthennarasu07@gmail.com",
    url="https://github.com/RahulThennarasu/cinder",
    packages=packages,  # Use our custom package list
    include_package_data=True,
    package_data={
        'backend.app': ['static/*', 'static/**/*'],
        'backend': ['.env'],
    },
    install_requires=[
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "pytorch": ["torch>=1.12.0"],
        "tensorflow": ["tensorflow>=2.8.0"],
        "firebase": ["firebase-admin>=5.0.0"],
        "all": [
            "torch>=1.12.0",
            "tensorflow>=2.8.0",
            "google-generativeai>=0.3.0",
            "firebase-admin>=5.0.0",
        ],
    },
    entry_points={
        'console_scripts': [
            'cinder=cinder.cli:main',
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)