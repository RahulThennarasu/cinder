#!/usr/bin/env python3
"""
CompileML Demo - Scikit-learn Classification

This script trains a scikit-learn model with intentional issues to demonstrate
CompileML's analysis capabilities:
- Class imbalance
- Underfitting (high bias)
- Feature importance analysis

Requirements:
- scikit-learn
- numpy
- CompileML (your backend code)
"""

import os
import sys
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import threading
import time

# Add the parent directory to the path to import CompileML
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import CompileML components
from backend.model_interface.connector import ModelDebugger

# Define a simple wrapper for the dataset compatible with CompileML
class SklearnDataset:
    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_samples = len(X)
        self.current = 0
    
    def __iter__(self):
        self.current = 0
        return self
    
    def __next__(self):
        if self.current >= self.n_samples:
            raise StopIteration
        
        end = min(self.current + self.batch_size, self.n_samples)
        X_batch = self.X[self.current:end]
        y_batch = self.y[self.current:end]
        self.current = end
        
        return X_batch, y_batch

def create_imbalanced_dataset():
    """Create a dataset with intentional class imbalance and some challenging characteristics."""
    print("Creating synthetic dataset with class imbalance...")
    
    # Create a base dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        class_sep=0.6,  # Make classes somewhat hard to separate
        random_state=42
    )
    
    # Create severe class imbalance (90% class 0, 10% class 1)
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]
    
    # Keep all of class 0 but only 10% of class 1
    keep_class_1 = np.random.choice(class_1_indices, size=int(len(class_1_indices) * 0.1), replace=False)
    keep_indices = np.concatenate([class_0_indices, keep_class_1])
    
    X_imbalanced = X[keep_indices]
    y_imbalanced = y[keep_indices]
    
    print(f"Dataset created:")
    print(f"  - Total samples: {len(X_imbalanced)}")
    print(f"  - Features: {X_imbalanced.shape[1]}")
    print(f"  - Class 0: {np.sum(y_imbalanced == 0)} samples ({np.mean(y_imbalanced == 0)*100:.1f}%)")
    print(f"  - Class 1: {np.sum(y_imbalanced == 1)} samples ({np.mean(y_imbalanced == 1)*100:.1f}%)")
    
    return X_imbalanced, y_imbalanced

def create_simple_model():
    """Create an intentionally simple model that will underfit."""
    # Use a very simple logistic regression that will struggle with the data
    model = LogisticRegression(
        C=0.01,  # High regularization to cause underfitting
        max_iter=50,  # Few iterations
        random_state=42,
        solver='liblinear'
    )
    return model

def create_better_model():
    """Create a more sophisticated model for comparison."""
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )
    return model

def train_and_analyze_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train a model and analyze it with CompileML."""
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    print(f"{'='*50}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
    
    # Create dataset wrapper for CompileML
    test_dataset = SklearnDataset(X_test, y_test)
    
    # Connect to CompileML for analysis
    print(f"\nConnecting {model_name} to CompileML...")
    debugger = ModelDebugger(model, test_dataset, name=f"{model_name}")
    
    # Add some mock training history for demonstration
    # In real usage, you'd collect this during actual training
    mock_history = []
    base_acc = max(0.5, accuracy - 0.2)  # Start lower than final accuracy
    for i in range(1, 11):
        # Simulate learning progress
        epoch_acc = min(accuracy, base_acc + (i * 0.03) + np.random.uniform(-0.01, 0.01))
        epoch_loss = max(0.1, 0.8 - (i * 0.07) + np.random.uniform(-0.02, 0.02))
        
        mock_history.append({
            "iteration": i,
            "accuracy": float(epoch_acc),
            "loss": float(epoch_loss),
            "learning_rate": 0.01 * (0.95 ** i)  # Decaying learning rate
        })
    
    debugger.training_history = mock_history
    
    # Run analysis
    print("Running CompileML analysis...")
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
    
    return debugger

def launch_dashboard_thread(debugger):
    """Launch dashboard in a separate thread."""
    debugger.launch_dashboard()

def main():
    """Main function to demonstrate scikit-learn with CompileML."""
    print("Scikit-learn Classification with CompileML")
    print("=" * 50)
    
    # Create dataset with intentional issues
    X, y = create_imbalanced_dataset()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Demo 1: Simple model (will show underfitting/high bias)
    simple_model = create_simple_model()
    debugger1 = train_and_analyze_model(
        simple_model, "Simple Logistic Regression", 
        X_train, X_test, y_train, y_test
    )
    
    # Demo 2: Better model (will show class imbalance issues)
    better_model = create_better_model()
    debugger2 = train_and_analyze_model(
        better_model, "Random Forest", 
        X_train, X_test, y_train, y_test
    )
    
    # Let user choose which model to analyze in the dashboard
    print(f"\n{'='*50}")
    print("Choose a model to analyze in the CompileML dashboard:")
    print("1. Simple Logistic Regression (shows underfitting)")
    print("2. Random Forest (shows class imbalance)")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                selected_debugger = debugger1
                model_name = "Simple Logistic Regression"
                break
            elif choice == '2':
                selected_debugger = debugger2
                model_name = "Random Forest"
                break
            elif choice == '3':
                print("Exiting...")
                return
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
    print(f"\nLaunching CompileML dashboard for {model_name}...")
    print("Access the dashboard at http://localhost:8000")
    print("Press Ctrl+C to exit")
    
    # Start the dashboard in a separate thread
    dashboard_thread = threading.Thread(
        target=launch_dashboard_thread,
        args=(selected_debugger,),
        daemon=False
    )
    dashboard_thread.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()