#!/usr/bin/env python3
"""
CompileML Demo - Customer Churn Prediction

This is a realistic machine learning example that demonstrates CompileML's
analysis capabilities on a common business problem: predicting customer churn.

This model intentionally includes several issues that CompileML can detect:
- Class imbalance (churn rate is typically 10-20%)
- Feature scaling issues
- Potential overfitting
- Missing regularization

Business Context:
A telecommunications company wants to predict which customers are likely 
to cancel their subscription (churn) so they can take proactive retention actions.

Dataset Features:
- Customer demographics (age, gender, senior citizen status)
- Service information (phone service, internet type, contract type)
- Account information (tenure, monthly charges, total charges)
- Target: Churn (Yes/No)

Requirements:
- pandas, numpy, scikit-learn
- CompileML (your backend code)
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to import CompileML
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import CompileML components
from backend.model_interface.connector import ModelDebugger

# Define a wrapper for the dataset compatible with CompileML
class ChurnDataset:
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

def create_synthetic_churn_data(n_samples=5000):
    """
    Create a realistic synthetic churn dataset that mimics real telecom data.
    This creates class imbalance and realistic feature relationships.
    """
    print("Creating synthetic customer churn dataset...")
    
    np.random.seed(42)  # For reproducibility
    
    # Customer Demographics
    ages = np.random.normal(40, 15, n_samples).clip(18, 80)
    is_senior = (ages >= 65).astype(int)
    gender = np.random.choice([0, 1], n_samples)  # 0: Female, 1: Male
    
    # Service Features
    has_phone = np.random.choice([0, 1], n_samples, p=[0.1, 0.9])
    has_internet = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    internet_type = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])  # 0: DSL, 1: Fiber, 2: No internet
    
    # Contract and Payment
    contract_type = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])  # 0: Month-to-month, 1: One year, 2: Two year
    paperless_billing = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    payment_method = np.random.choice([0, 1, 2, 3], n_samples)  # Different payment methods
    
    # Account Information
    tenure_months = np.random.exponential(20, n_samples).clip(1, 72)
    monthly_charges = 20 + has_phone * 10 + has_internet * 30 + internet_type * 10 + np.random.normal(0, 10, n_samples)
    monthly_charges = monthly_charges.clip(20, 120)
    total_charges = monthly_charges * tenure_months + np.random.normal(0, 100, n_samples)
    total_charges = total_charges.clip(0, None)
    
    # Create realistic churn probability based on features
    # Higher churn probability for:
    # - Month-to-month contracts
    # - Higher monthly charges
    # - Lower tenure
    # - Senior citizens
    # - Fiber internet (higher expectations)
    
    churn_prob = (
        0.05 +  # Base churn rate
        (contract_type == 0) * 0.25 +  # Month-to-month penalty
        (monthly_charges > 70) * 0.15 +  # High charges penalty
        (tenure_months < 6) * 0.20 +  # New customer penalty
        is_senior * 0.10 +  # Senior citizen penalty
        (internet_type == 1) * 0.08 +  # Fiber internet penalty
        np.random.normal(0, 0.1, n_samples)  # Random noise
    )
    
    churn_prob = np.clip(churn_prob, 0, 0.8)  # Realistic churn rates
    churn = np.random.binomial(1, churn_prob, n_samples)
    
    # Create DataFrame
    data = {
        'Age': ages,
        'Gender': gender,
        'SeniorCitizen': is_senior,
        'HasPhone': has_phone,
        'HasInternet': has_internet,
        'InternetType': internet_type,
        'ContractType': contract_type,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'TenureMonths': tenure_months,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    }
    
    df = pd.DataFrame(data)
    
    # Print dataset statistics
    print(f"Dataset created with {len(df)} customers:")
    print(f"  - Churn rate: {df['Churn'].mean()*100:.1f}%")
    print(f"  - Average age: {df['Age'].mean():.1f} years")
    print(f"  - Average tenure: {df['TenureMonths'].mean():.1f} months")
    print(f"  - Average monthly charges: ${df['MonthlyCharges'].mean():.2f}")
    
    return df

def preprocess_data(df):
    """
    Preprocess the churn dataset for machine learning.
    Intentionally skip some preprocessing steps to create issues for CompileML to detect.
    """
    print("Preprocessing data...")
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Convert to numpy arrays
    X = X.values
    y = y.values
    
    # NOTE: Intentionally NOT scaling the data to create issues for CompileML to detect
    # In a real scenario, you'd want to scale features like MonthlyCharges and TotalCharges
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)} (Class 0: No Churn, Class 1: Churn)")
    
    return X, y

def create_problematic_model():
    """
    Create a model with intentional issues for CompileML to detect:
    - No regularization
    - High complexity (prone to overfitting)
    - No class balancing
    """
    # Use a Random Forest with high complexity
    model = RandomForestClassifier(
        n_estimators=200,       # Many trees
        max_depth=None,         # No depth limit (can overfit)
        min_samples_split=2,    # Minimum splits (can overfit)
        min_samples_leaf=1,     # Minimum leaf samples (can overfit)
        # class_weight=None,    # Not handling class imbalance
        random_state=42
    )
    return model, "Overly Complex Random Forest"

def create_better_model():
    """
    Create a better model for comparison.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,           # Limit depth
        min_samples_split=10,   # Require more samples to split
        min_samples_leaf=5,     # Require more samples in leaves
        class_weight='balanced', # Handle class imbalance
        random_state=42
    )
    return model, "Balanced Random Forest"

def train_and_analyze_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train a model and analyze it with CompileML."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['No Churn', 'Churn']))
    
    # Show confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                No    Churn")
    print(f"Actual No    {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       Churn {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        feature_names = ['Age', 'Gender', 'SeniorCitizen', 'HasPhone', 'HasInternet', 
                        'InternetType', 'ContractType', 'PaperlessBilling', 
                        'PaymentMethod', 'TenureMonths', 'MonthlyCharges', 'TotalCharges']
        
        importances = model.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 5 Most Important Features:")
        for feature, importance in feature_importance[:5]:
            print(f"  {feature}: {importance:.4f}")
    
    # Create dataset wrapper for CompileML
    test_dataset = ChurnDataset(X_test, y_test)
    
    # Connect to CompileML for analysis
    print(f"\nConnecting {model_name} to CompileML...")
    debugger = ModelDebugger(model, test_dataset, name=model_name)
    
    # Add mock training history for demonstration
    training_history = []
    base_acc = max(0.6, accuracy - 0.15)
    for i in range(1, 21):  # 20 epochs
        # Simulate realistic training progress
        epoch_acc = min(0.95, base_acc + (i * 0.015) + np.random.uniform(-0.01, 0.01))
        epoch_loss = max(0.1, 0.7 - (i * 0.025) + np.random.uniform(-0.02, 0.02))
        
        training_history.append({
            "iteration": i,
            "accuracy": float(epoch_acc),
            "loss": float(epoch_loss),
            "learning_rate": 0.01 * (0.95 ** (i // 5))  # Decay every 5 epochs
        })
    
    debugger.training_history = training_history
    
    # Run analysis
    print("Running CompileML analysis...")
    results = debugger.analyze()
    
    # Print analysis results
    print(f"\nCompileML Analysis Results:")
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

def main():
    """Main function to demonstrate customer churn prediction with CompileML."""
    print("Customer Churn Prediction with CompileML")
    print("=" * 60)
    print("Business Problem: Predict which customers are likely to cancel their subscription")
    print("This helps the company take proactive retention actions.")
    print()
    
    # Create synthetic churn dataset
    df = create_synthetic_churn_data(n_samples=3000)  # Smaller for faster demo
    
    # Preprocess the data
    X, y = preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} customers")
    print(f"Test set: {X_test.shape[0]} customers")
    print(f"Features: {X_train.shape[1]}")
    
    # Train multiple models for comparison
    print(f"\n{'='*60}")
    print("TRAINING MODELS FOR COMPARISON")
    print(f"{'='*60}")
    
    # Model 1: Problematic model (overfitting, class imbalance)
    problematic_model, problematic_name = create_problematic_model()
    debugger1 = train_and_analyze_model(
        problematic_model, problematic_name, 
        X_train, X_test, y_train, y_test
    )
    
    # Model 2: Better model (balanced)
    better_model, better_name = create_better_model()
    debugger2 = train_and_analyze_model(
        better_model, better_name, 
        X_train, X_test, y_train, y_test
    )
    
    # Let user choose which model to analyze in the dashboard
    print(f"\n{'='*60}")
    print("Choose a model to analyze in the CompileML dashboard:")
    print("1. Overly Complex Random Forest (shows overfitting & class imbalance)")
    print("2. Balanced Random Forest (shows better practices)")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                selected_debugger = debugger1
                model_name = problematic_name
                break
            elif choice == '2':
                selected_debugger = debugger2
                model_name = better_name
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
    print("\nKey insights you'll see:")
    print("- Class imbalance analysis (churn rate ~15%)")
    print("- Feature importance for business decisions")
    print("- Model complexity vs. performance trade-offs")
    print("- Suggestions for handling imbalanced datasets")
    print("- Training history and learning curves")
    print("\nPress Ctrl+C to exit")
    
    # Start the dashboard
    try:
        selected_debugger.launch_dashboard()
        # Keep the main thread alive
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()