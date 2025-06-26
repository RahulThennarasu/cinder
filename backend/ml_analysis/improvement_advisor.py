import numpy as np
import torch
from typing import Dict, Any, List, Optional
import logging

class ModelImprovementAdvisor:
    """
    Analyzes model performance and provides actionable improvement suggestions.
    """
    
    def __init__(self, model_debugger):
        """
        Initialize with a ModelDebugger instance.
        
        Args:
            model_debugger: Instance of ModelDebugger
        """
        self.debugger = model_debugger
        
    def suggest_improvements(self, detail_level="comprehensive") -> Dict[str, Any]:
        """
        Generate specific, actionable suggestions to improve the model.
        
        Args:
            detail_level: Level of detail for suggestions ('basic', 'comprehensive', 'code')
        
        Returns:
            Dict with categorized improvement suggestions
        """
        # Make sure we have analysis results
        if self.debugger.predictions is None or self.debugger.ground_truth is None:
            self.debugger.analyze()
        
        # Get key metrics
        accuracy = self.debugger._calculate_accuracy()
        error_analysis = self.debugger._analyze_errors()
        
        suggestions = []
        
        # Check for class imbalance
        class_distribution = np.bincount(self.debugger.ground_truth)
        major_class = np.argmax(class_distribution)
        major_class_ratio = class_distribution[major_class] / len(self.debugger.ground_truth)
        
        if major_class_ratio > 0.7:  # Significant imbalance
            suggestions.append({
                "category": "data_preparation",
                "title": "Address Class Imbalance",
                "issue": f"Class imbalance detected with {major_class_ratio*100:.1f}% of samples in class {major_class}",
                "suggestion": "Use class weighting or resampling techniques to balance your dataset",
                "severity": "high",
                "expected_impact": "medium-high",
                "code_example": self._generate_code_example("class_imbalance") if detail_level == "code" else None
            })
        
        # Check for high bias
        if accuracy < 0.75:
            suggestions.append({
                "category": "model_capacity",
                "title": "Increase Model Complexity",
                "issue": f"Possible underfitting (high bias) with accuracy of {accuracy*100:.1f}%",
                "suggestion": "Increase model complexity, add more features, or reduce regularization",
                "severity": "high",
                "expected_impact": "high",
                "code_example": self._generate_code_example("increase_complexity") if detail_level == "code" else None
            })
        
        # Check for overfitting
        if hasattr(self.debugger, 'training_history') and len(self.debugger.training_history) > 0:
            train_acc = self.debugger.training_history[-1].get('accuracy', 0)
            val_acc = accuracy
            if train_acc - val_acc > 0.15:  # Significant gap
                suggestions.append({
                    "category": "generalization",
                    "title": "Add Regularization",
                    "issue": f"Possible overfitting (high variance) with training accuracy {train_acc*100:.1f}% vs. validation accuracy {val_acc*100:.1f}%",
                    "suggestion": "Add regularization, use more training data, or simplify the model",
                    "severity": "high",
                    "expected_impact": "high",
                    "code_example": self._generate_code_example("regularization") if detail_level == "code" else None
                })
        
        # Check for feature importance
        if hasattr(self.debugger, '_feature_importances') and self.debugger._feature_importances is not None:
            top_importance = np.max(self.debugger._feature_importances)
            if top_importance > 0.5:  # One feature dominates
                suggestions.append({
                    "category": "feature_engineering",
                    "title": "Improve Feature Balance",
                    "issue": "Model relies too heavily on a single feature",
                    "suggestion": "Add more informative features or use feature selection techniques",
                    "severity": "medium",
                    "expected_impact": "medium-high",
                    "code_example": self._generate_code_example("feature_engineering") if detail_level == "code" else None
                })
        
        # Always suggest cross-validation for reliable evaluation
        suggestions.append({
            "category": "evaluation",
            "title": "Use Cross-Validation",
            "issue": "Single train-test split may not be reliable",
            "suggestion": "Use k-fold cross-validation for more robust performance evaluation",
            "severity": "medium",
            "expected_impact": "medium",
            "code_example": self._generate_code_example("cross_validation") if detail_level == "code" else None
        })
        
        # Add hyperparameter tuning suggestion
        suggestions.append({
            "category": "optimization",
            "title": "Tune Hyperparameters",
            "issue": "Default hyperparameters may not be optimal",
            "suggestion": "Perform systematic hyperparameter tuning",
            "severity": "medium",
            "expected_impact": "medium-high",
            "code_example": self._generate_code_example("hyperparameter_tuning") if detail_level == "code" else None
        })
        
        # Suggest ensemble methods if applicable
        if accuracy < 0.95:
            suggestions.append({
                "category": "advanced_techniques",
                "title": "Try Ensemble Methods",
                "issue": "Single models often have limitations in performance",
                "suggestion": "Combine multiple models using ensemble techniques",
                "severity": "low",
                "expected_impact": "medium-high",
                "code_example": self._generate_code_example("ensemble_methods") if detail_level == "code" else None
            })
        
        return {
            "model_accuracy": accuracy,
            "error_rate": error_analysis['error_rate'],
            "improvement_potential": "high" if accuracy < 0.9 else "medium" if accuracy < 0.95 else "low",
            "suggestions": suggestions
        }
    
    def _generate_code_example(self, category: str) -> Dict[str, str]:
        """
        Generate framework-specific code examples for suggested improvements.
        
        Args:
            category: The category of suggestion
            
        Returns:
            Dict with code examples for different frameworks
        """
        framework = self.debugger.framework.lower()
        examples = {}
        
        # Get the appropriate code example based on framework and category
        if category == "class_imbalance":
            if framework == "pytorch":
                examples["pytorch"] = """
# PyTorch: Handle class imbalance
import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler

# Calculate class weights (inverse frequency)
y_train = # your training labels here
class_counts = torch.bincount(torch.tensor(y_train))
class_weights = 1.0 / class_counts.float()
sample_weights = torch.tensor([class_weights[t] for t in y_train])

# Create a weighted sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(y_train),
    replacement=True
)

# Use in DataLoader
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=32,
    sampler=sampler
)

# Alternative: Use weighted loss function
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
"""
            elif framework == "tensorflow":
                examples["tensorflow"] = """
# TensorFlow: Handle class imbalance
import tensorflow as tf
import numpy as np

# Calculate class weights
y_train = # your training labels here
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = {}

for class_idx, count in enumerate(class_counts):
    class_weights[class_idx] = total_samples / (len(class_counts) * count)

# Method 1: Use in model.fit()
model.fit(
    X_train, y_train,
    class_weight=class_weights,
    batch_size=32,
    epochs=10,
    validation_split=0.2
)

# Method 2: Use SMOTE for oversampling
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train on balanced dataset
model.fit(
    X_resampled, y_resampled,
    batch_size=32,
    epochs=10,
    validation_split=0.2
)
"""
            else:  # sklearn
                examples["sklearn"] = """
# Scikit-learn: Handle class imbalance
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Method 1: Use class weights
y_train = # your training labels here
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Convert to dictionary for most models
weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Apply to model (example with Random Forest)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,
    class_weight=weights_dict,
    random_state=42
)

# Method 2: Use SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train on balanced dataset
model.fit(X_resampled, y_resampled)
"""
        
        elif category == "regularization":
            if framework == "pytorch":
                examples["pytorch"] = """
# PyTorch: Add regularization to prevent overfitting
import torch.nn as nn
import torch.optim as optim

# Method 1: Add dropout layers to model
class RegularizedModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RegularizedModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.5)  # Add dropout with 0.5 probability
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(0.3)  # Add dropout with 0.3 probability
        self.layer3 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.dropout1(x)  # Apply dropout
        x = self.layer2(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)  # Apply dropout
        x = self.layer3(x)
        return x

# Method 2: Use weight decay (L2 regularization)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Method 3: Implement early stopping
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(100):  # Training loop
    # Train for one epoch...
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    
    # Evaluate on validation set
    val_loss = validate(model, val_loader, criterion)
    
    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        # Load best model
        model.load_state_dict(torch.load('best_model.pth'))
        break
"""
            elif framework == "tensorflow":
                examples["tensorflow"] = """
# TensorFlow: Add regularization to prevent overfitting
import tensorflow as tf

# Method 1: Add regularization to layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        128, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),  # L2 regularization
        input_shape=(input_shape,)
    ),
    tf.keras.layers.Dropout(0.5),  # Add dropout layer
    tf.keras.layers.Dense(
        64, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Method 2: Use early stopping and model checkpointing
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True  # Restore model to best weights when stopped
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Use callbacks in training
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint]
)
"""
            else:  # sklearn
                examples["sklearn"] = """
# Scikit-learn: Add regularization to prevent overfitting

# Method 1: For linear models (Logistic Regression)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=0.1,  # Inverse of regularization strength (smaller value = stronger regularization)
    penalty='l2',  # L2 regularization
    solver='liblinear',
    max_iter=1000,
    random_state=42
)

# Method 2: For tree-based models (Random Forest)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,           # Limit tree depth
    min_samples_split=5,    # Require more samples to split a node
    min_samples_leaf=2,     # Require more samples in leaf nodes
    max_features='sqrt',    # Use only a subset of features for each split
    random_state=42
)

# Method 3: For neural networks
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    alpha=0.01,             # L2 penalty parameter
    early_stopping=True,    # Use early stopping
    validation_fraction=0.2,
    n_iter_no_change=10,    # Number of iterations with no improvement
    random_state=42
)

# Train the model
model.fit(X_train, y_train)
"""
            
        # Add more categories as needed...
        
        # If we don't have a specific example for this framework, use sklearn as fallback
        if framework not in examples and "sklearn" in examples:
            examples[framework] = examples["sklearn"]
        
        # If we don't have any examples for this category, provide a generic message
        if not examples:
            examples[framework] = "# Code example not available for this category and framework."
        
        return examples