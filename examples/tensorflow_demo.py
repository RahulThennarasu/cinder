#!/usr/bin/env python3
"""
CompileML Demo - TensorFlow/Keras Classification

This script trains a TensorFlow model with intentional issues to demonstrate
CompileML's analysis capabilities:
- Overfitting (high variance)
- Class imbalance
- Training history tracking

Requirements:
- tensorflow
- numpy
- CompileML (your backend code)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import threading
import time

# Add the parent directory to the path to import CompileML
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import CompileML components
from backend.model_interface.connector import ModelDebugger

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class TensorFlowDataset:
    """Wrapper to make TensorFlow data compatible with CompileML."""
    
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

def create_complex_dataset():
    """Create a dataset that will show overfitting with a complex model."""
    print("Creating synthetic dataset for TensorFlow demo...")
    
    # Create a moderately complex dataset
    X, y = make_classification(
        n_samples=800,  # Smaller dataset to encourage overfitting
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        class_sep=0.8,
        random_state=42
    )
    
    # Create class imbalance (75% class 0, 25% class 1)
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]
    
    # Keep all of class 0 but only 33% of class 1 (to get ~75/25 split)
    keep_class_1 = np.random.choice(class_1_indices, size=int(len(class_1_indices) * 0.33), replace=False)
    keep_indices = np.concatenate([class_0_indices, keep_class_1])
    
    X_imbalanced = X[keep_indices]
    y_imbalanced = y[keep_indices]
    
    print(f"Dataset created:")
    print(f"  - Total samples: {len(X_imbalanced)}")
    print(f"  - Features: {X_imbalanced.shape[1]}")
    print(f"  - Class 0: {np.sum(y_imbalanced == 0)} samples ({np.mean(y_imbalanced == 0)*100:.1f}%)")
    print(f"  - Class 1: {np.sum(y_imbalanced == 1)} samples ({np.mean(y_imbalanced == 1)*100:.1f}%)")
    
    return X_imbalanced, y_imbalanced

def create_overfitting_model(input_shape):
    """Create an overly complex model that will overfit."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Use a high learning rate to make overfitting more obvious
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_regularized_model(input_shape):
    """Create a better model with regularization."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class TrainingHistoryCallback(tf.keras.callbacks.Callback):
    """Custom callback to collect training history for CompileML."""
    
    def __init__(self):
        super().__init__()
        self.history_data = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.history_data.append({
            "iteration": epoch + 1,
            "accuracy": float(logs.get('val_accuracy', logs.get('accuracy', 0))),
            "loss": float(logs.get('val_loss', logs.get('loss', 0))),
            "learning_rate": float(self.model.optimizer.learning_rate)
        })

def train_and_analyze_model(model, model_name, X_train, X_test, y_train, y_test, epochs=20):
    """Train a TensorFlow model and analyze it with CompileML."""
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    print(f"{'='*50}")
    
    # Create training history callback
    history_callback = TrainingHistoryCallback()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[history_callback],
        verbose=1
    )
    
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\n{model_name} Test Accuracy: {accuracy:.4f}")
    
    # Print some training vs validation metrics to show overfitting
    if 'val_accuracy' in history.history:
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Overfitting Gap: {(final_train_acc - final_val_acc)*100:.1f}%")
    
    # Create dataset wrapper for CompileML
    test_dataset = TensorFlowDataset(X_test, y_test)
    
    # Connect to CompileML for analysis
    print(f"\nConnecting {model_name} to CompileML...")
    debugger = ModelDebugger(model, test_dataset, name=f"{model_name}")
    
    # Add the collected training history
    debugger.training_history = history_callback.history_data
    
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
    if 'roc_auc' in results:
        print(f"- ROC AUC: {results['roc_auc']*100:.2f}%")
    print(f"- Errors: {results['error_analysis']['error_count']} ({results['error_analysis']['error_rate']*100:.2f}%)")
    
    return debugger

def launch_dashboard_thread(debugger):
    """Launch dashboard in a separate thread."""
    debugger.launch_dashboard()

def main():
    """Main function to demonstrate TensorFlow with CompileML."""
    print("TensorFlow/Keras Classification with CompileML")
    print("=" * 50)
    
    # Create dataset
    X, y = create_complex_dataset()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize the features (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    print(f"Features: {X_train_scaled.shape[1]}")
    
    # Demo 1: Overfitting model (complex architecture, high learning rate)
    print("\n" + "="*60)
    print("DEMO 1: Overly Complex Model (will show overfitting)")
    print("="*60)
    
    overfitting_model = create_overfitting_model(X_train_scaled.shape[1])
    debugger1 = train_and_analyze_model(
        overfitting_model, "Complex Neural Network", 
        X_train_scaled, X_test_scaled, y_train, y_test, epochs=30
    )
    
    # Demo 2: Regularized model (better architecture with dropout)
    print("\n" + "="*60)
    print("DEMO 2: Regularized Model (better generalization)")
    print("="*60)
    
    regularized_model = create_regularized_model(X_train_scaled.shape[1])
    debugger2 = train_and_analyze_model(
        regularized_model, "Regularized Neural Network", 
        X_train_scaled, X_test_scaled, y_train, y_test, epochs=25
    )
    
    # Let user choose which model to analyze in the dashboard
    print(f"\n{'='*50}")
    print("Choose a model to analyze in the CompileML dashboard:")
    print("1. Complex Neural Network (shows overfitting)")
    print("2. Regularized Neural Network (shows class imbalance)")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                selected_debugger = debugger1
                model_name = "Complex Neural Network"
                break
            elif choice == '2':
                selected_debugger = debugger2
                model_name = "Regularized Neural Network"
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
    print("\nKey features to explore in the dashboard:")
    print("- Training History tab: See accuracy/loss curves over time")
    print("- Learning Rate tab: Monitor learning rate changes")
    print("- Class Distribution: Analyze prediction patterns")
    print("- Model Improvement suggestions for fixing issues")
    print("\nPress Ctrl+C to exit")
    
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