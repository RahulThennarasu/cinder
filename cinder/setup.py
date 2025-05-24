from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cinder-ml",  # Use 'cinder-ml' since 'cinder' might be taken
    version="0.1.0",
    author="Rahul Thennarasu",
    author_email="rahulthennarasu07@gmail.com",
    description="A tool for ML model debugging and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RahulThennarasu/cinder",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "torch",
        "tensorflow",
        "scikit-learn",
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "google-generativeai",  # For Gemini integration
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "cinder-serve=cinder.app.server:start_server_cli",
        ],
    },
)