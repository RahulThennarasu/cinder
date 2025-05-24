#!/usr/bin/env python3
"""
CompileML Demo - Tabular Data Classification with scikit-learn

This script trains several scikit-learn models on the Breast Cancer Wisconsin dataset
and connects them to CompileML for analysis and debugging.

Requirements:
- scikit-learn
- numpy
- matplotlib (optional, for plotting)
- CompileML (your backend code)
"""

import os
import sys
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Add the parent directory to the path to import CompileML
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import CompileML components
from cinder.cinder.model_interface.connector import ModelDebugger

# Define a simple wrapper for the dataset compatible with CompileML
class SklearnDataset:
    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_samples = len(X)
        self.current = 0  # Initialize current counter
    
    def __iter__(self):
        self.current = 0  # Reset the counter when starting iteration
        return self
    
    def __next__(self):
        if self.current >= self.n_samples:
            raise StopIteration
        
        end = min(self.current + self.batch_size, self.n_samples)
        X_batch = self.X[self.current:end]
        y_batch = self.y[self.current:end]
        self.current = end
        
        return X_batch, y_batch

def main():
    """Main function to train and analyze scikit-learn models."""
    print("Breast Cancer Classification with CompileML")
    print("=" * 50)
    
    # Load the breast cancer dataset
    print("Loading dataset...")
    data = load_breast_cancer()
    # Access as attributes instead of dictionary items to avoid type checker errors
    X = data.data  # This is equivalent to data['data']
    y = data.target  # This is equivalent to data['target']
    feature_names = data.feature_names  # This is equivalent to data['feature_names']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Feature names: {feature_names[:5]}... (total: {len(feature_names)})")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Standardize the features
    print("Preprocessing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create a wrapper for the test data compatible with CompileML
    test_dataset = SklearnDataset(X_test_scaled, y_test)
    
    # Define models to evaluate
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    # Train and evaluate the models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Print classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Show detailed classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['benign', 'malignant']))
        
        # Connect to CompileML for analysis
        print(f"Connecting {name} to CompileML...")
        debugger = ModelDebugger(model, test_dataset, name=f"{name} Classifier")
        
        # Run analysis
        results = debugger.analyze()
        
        # Print analysis results
        print("\nCompileML Analysis Results:")
        print(f"- Model: {results['model_name']} ({results['framework']})")
        print(f"- Dataset size: {results['dataset_size']} samples")
        print(f"- Accuracy: {results['accuracy']*100:.2f}%")
        if 'precision' in results:
            print(f"- Precision: {results['precision']*100:.2f}%")
        if 'recall' in results:
            print(f"- Recall: {results['recall']*100:.2f}%")
        if 'f1' in results:
            print(f"- F1 Score: {results['f1']*100:.2f}%")
        print(f"- Errors: {results['error_analysis']['error_count']} ({results['error_analysis']['error_rate']*100:.2f}%)")
        
        # Ask user if they want to launch the dashboard for this model
        response = input(f"\nLaunch CompileML dashboard for {name}? (y/n): ")
        if response.lower() == 'y':
            print(f"\nLaunching CompileML dashboard for {name}...")
            print("Access the dashboard at http://localhost:8000")
            print("Press Ctrl+C to exit")
            debugger.launch_dashboard()

            import time
            try:
                while True:
                    time.sleep(1)  # Sleep to prevent high CPU usage
            except KeyboardInterrupt:
                print("\nShutting down...")
            break


if __name__ == "__main__":
    main()