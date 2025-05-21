import os
import requests
import json
import time
from typing import Dict, Any, Optional
import hashlib

class SimpleCodeGenerator:
    """Code example generator using Gemini API with caching to avoid rate limits."""
    
    def __init__(self, api_key=None):
        """Initialize with Gemini API key."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            print("Warning: No Gemini API key provided. Code generation will not be available.")
        
        # Create a cache directory if it doesn't exist
        os.makedirs('code_cache', exist_ok=True)
        
        # Keep track of when we last called the API to avoid rate limits
        self.last_api_call = 0
        self.min_call_interval = 5  # seconds between API calls
    
    def generate_code_example(self, 
                             framework: str, 
                             category: str, 
                             model_context: Dict[str, Any]) -> str:
        """Generate code example for ML model improvement using Gemini API with caching."""
        if not self.api_key:
            return "# Code example generation unavailable - API key not configured"
        
        # Create a cache key based on the inputs
        cache_key = self._create_cache_key(framework, category, model_context)
        cache_file = os.path.join('code_cache', f"{cache_key}.py")
        
        # Check if we have this cached
        if os.path.exists(cache_file):
            print(f"Using cached code for {framework} - {category}")
            with open(cache_file, 'r') as f:
                return f.read()
            
        # Provide fallback templates for common categories to reduce API calls
        fallback_code = self._get_fallback_code(framework, category)
        if fallback_code:
            print(f"Using fallback template for {framework} - {category}")
            # Save to cache
            with open(cache_file, 'w') as f:
                f.write(fallback_code)
            return fallback_code
        
        # Check for rate limiting
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.min_call_interval:
            wait_time = self.min_call_interval - time_since_last_call
            print(f"Rate limiting: waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        try:
            # Update last API call time
            self.last_api_call = time.time()
            
            # Prepare the prompt with detailed context
            accuracy = model_context.get('accuracy', 0)
            error_rate = model_context.get('error_rate', 0)
            
            framework_name = {
                'pytorch': 'PyTorch',
                'tensorflow': 'TensorFlow',
                'sklearn': 'scikit-learn'
            }.get(framework, framework)
            
            category_display = category.replace('_', ' ').title()
            
            # Create a more detailed prompt for better results
            prompt = f"""
            As an expert ML developer, write a complete, production-ready {framework_name} implementation for {category_display}.
            
            Model details:
            - Current accuracy: {accuracy:.4f}
            - Error rate: {error_rate:.4f}
            - Framework: {framework_name}
            
            The code should:
            - Be well-organized and follow best practices
            - Include proper comments and docstrings
            - Be ready to run with minimal modifications
            - Handle edge cases appropriately
            
            Return ONLY the executable code without additional explanations before or after.
            """
            
            # Call the Gemini API
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.api_key}"
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 4096,
                    "topP": 0.95
                }
            }
            
            # Send the request
            response = requests.post(api_url, headers=headers, json=data)
            
            # Check for rate limiting and other errors
            if response.status_code == 429:
                print("Rate limit exceeded for Gemini API")
                # Return fallback if available
                if fallback_code:
                    return fallback_code
                return "# Rate limit exceeded for Gemini API. Please try again later."
                
            if response.status_code != 200:
                error_message = f"API Error (Code: {response.status_code}): {response.text}"
                print(error_message)
                # Return fallback if available
                if fallback_code:
                    return fallback_code
                return f"# {error_message}\n\n# Code generation failed. Please check your API key."
            
            # Parse the successful response
            result = response.json()
            
            # Extract the generated code from the response
            if ('candidates' in result and 
                len(result['candidates']) > 0 and 
                'content' in result['candidates'][0] and 
                'parts' in result['candidates'][0]['content'] and 
                len(result['candidates'][0]['content']['parts']) > 0):
                
                code = result['candidates'][0]['content']['parts'][0]['text']
                
                # Clean up the code (remove markdown code block markers if present)
                code = code.replace("```python", "").replace("```", "").strip()
                
                # Save to cache
                with open(cache_file, 'w') as f:
                    f.write(code)
                
                return code
            else:
                print("Unexpected API response format:", json.dumps(result, indent=2))
                return "# Could not parse Gemini API response. Please try again."
                
        except Exception as e:
            error_message = f"Error generating code: {str(e)}"
            print(error_message)
            return f"# {error_message}"
    
    def _create_cache_key(self, framework, category, model_context):
        """Create a unique cache key based on the inputs."""
        # Combine inputs into a string
        key_str = f"{framework}_{category}_acc{model_context.get('accuracy', 0):.2f}_err{model_context.get('error_rate', 0):.2f}"
        
        # Create a hash of the string
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_fallback_code(self, framework, category):
        """Provide some fallback templates for common categories."""
        templates = {}
        
        # Cross-validation templates
        if category == "use_cross_validation" or category == "cross_validation":
            templates["pytorch"] = """# PyTorch Cross-Validation Implementation
import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader

def cross_validate(model, dataset, batch_size=32, num_epochs=10, k_folds=5, learning_rate=0.001):
    """
    Perform k-fold cross-validation on a PyTorch model.
    
    Args:
        model: PyTorch model class (will be instantiated fresh for each fold)
        dataset: PyTorch Dataset object
        batch_size: Batch size for training
        num_epochs: Number of training epochs per fold
        k_folds: Number of folds
        learning_rate: Learning rate for optimizer
        
    Returns:
        List of validation accuracies for each fold
    """
    # Set up k-fold cross validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # For storing accuracy results
    fold_results = []
    
    # K-Fold Cross Validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold+1}/{k_folds}')
        
        # Sample elements randomly from a given list of indices, no replacement
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        
        # Define data loaders for training and validation
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        
        # Initialize a fresh model for each fold
        model_instance = model()
        
        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            # Set model to training mode
            model_instance.train()
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model_instance(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            # Print epoch results
            if epoch == num_epochs - 1 or epoch % 10 == 0:
                # Set model to evaluation mode
                model_instance.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                # No gradient tracking needed for evaluation
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = model_instance(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        
                        # Get predictions
                        _, predicted = torch.max(outputs, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                
                # Calculate validation accuracy
                accuracy = correct / total
                print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {accuracy:.4f}')
        
        # Final evaluation on validation set
        model_instance.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model_instance(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # Calculate and store fold results
        fold_accuracy = correct / total
        print(f'Fold {fold+1} Accuracy: {fold_accuracy:.4f}')
        fold_results.append(fold_accuracy)
    
    # Print overall results
    print(f'K-Fold Cross Validation Results:')
    for i, result in enumerate(fold_results):
        print(f'Fold {i+1}: {result:.4f}')
    print(f'Average: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}')
    
    return fold_results"""
            
            templates["tensorflow"] = """# TensorFlow/Keras Cross-Validation Implementation
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold

def cross_validate(model_fn, X, y, batch_size=32, epochs=10, k_folds=5):
    """
    Perform k-fold cross-validation on a Keras model.
    
    Args:
        model_fn: Function that returns a compiled Keras model
        X: Input features
        y: Target labels
        batch_size: Batch size for training
        epochs: Number of epochs for training
        k_folds: Number of folds for cross-validation
    
    Returns:
        List of validation accuracies for each fold
    """
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Store the validation scores for each fold
    fold_accuracies = []
    
    # K-fold Cross Validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f'Training for fold {fold+1}/{k_folds}')
        
        # Create training and validation sets for the current fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create a fresh model for each fold
        model = model_fn()
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model on the validation data
        _, accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracies.append(accuracy)
        
        print(f'Fold {fold+1} - Validation Accuracy: {accuracy:.4f}')
    
    # Print average validation accuracy
    print('--------------------------------------------')
    print(f'Average Validation Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}')
    
    return fold_accuracies"""
            
            templates["sklearn"] = """# scikit-learn Cross-Validation Implementation
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import clone
import numpy as np
import pandas as pd

def simple_cross_validation(model, X, y, cv=5, scoring='accuracy'):
    """
    Perform simple cross-validation using scikit-learn's built-in function.
    
    Args:
        model: A scikit-learn estimator
        X: Input features
        y: Target labels
        cv: Number of folds (default=5)
        scoring: Scoring metric ('accuracy', 'f1', etc.)
        
    Returns:
        Cross-validation scores
    """
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    # Print results
    print(f"{cv}-Fold Cross Validation Results:")
    for i, score in enumerate(cv_scores):
        print(f"Fold {i+1}: {score:.4f}")
    print(f"Average: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return cv_scores

def detailed_cross_validation(model, X, y, cv=5):
    """
    Perform detailed cross-validation with multiple metrics.
    
    Args:
        model: A scikit-learn estimator
        X: Input features
        y: Target labels
        cv: Number of folds (default=5)
        
    Returns:
        DataFrame with detailed cross-validation results
    """
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro'
    }
    
    # Perform cross-validation with multiple metrics
    cv_results = cross_validate(
        model, X, y, 
        cv=cv, 
        scoring=scoring,
        return_train_score=True,
        return_estimator=True
    )
    
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'Fold': range(1, cv+1),
        'Train Accuracy': cv_results['train_accuracy'],
        'Test Accuracy': cv_results['test_accuracy'],
        'Train Precision': cv_results['train_precision_macro'],
        'Test Precision': cv_results['test_precision_macro'],
        'Train Recall': cv_results['train_recall_macro'],
        'Test Recall': cv_results['test_recall_macro'],
        'Train F1': cv_results['train_f1_macro'],
        'Test F1': cv_results['test_f1_macro']
    })
    
    # Print fold results
    print(summary)
    
    # Print average results
    means = summary.mean(numeric_only=True)
    stds = summary.std(numeric_only=True)
    print("\nAverage Results:")
    for metric in means.index:
        if metric != 'Fold':
            print(f"{metric}: {means[metric]:.4f} ± {stds[metric]:.4f}")
    
    return summary, cv_results['estimator']

def manual_cross_validation(model, X, y, cv=5):
    """
    Perform manual k-fold cross-validation with detailed reporting.
    
    Args:
        model: A scikit-learn estimator
        X: Input features
        y: Target labels
        cv: Number of folds (default=5)
        
    Returns:
        List of trained models, one for each fold
    """
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Lists to store results
    fold_models = []
    fold_scores = []
    fold_reports = []
    fold_matrices = []
    
    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold+1}/{cv}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Clone the model
        model_clone = clone(model)
        
        # Train the model
        model_clone.fit(X_train, y_train)
        
        # Predict
        y_pred = model_clone.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)
        matrix = confusion_matrix(y_val, y_pred)
        
        # Store results
        fold_models.append(model_clone)
        fold_scores.append(accuracy)
        fold_reports.append(report)
        fold_matrices.append(matrix)
        
        # Print results
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(classification_report(y_val, y_pred))
    
    # Print final results
    print("\nCross-Validation Results:")
    for i, score in enumerate(fold_scores):
        print(f"Fold {i+1}: {score:.4f}")
    print(f"Average: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    return fold_models, fold_scores, fold_reports, fold_matrices

# Example usage:
# scores = simple_cross_validation(model, X, y)
# summary, models = detailed_cross_validation(model, X, y)
# models, scores, reports, matrices = manual_cross_validation(model, X, y)"""
        
        # Add more templates for other common categories as needed
        
        # Return the template if it exists
        if framework in templates:
            return templates[framework]
        
        return None