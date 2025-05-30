import logging
from typing import Dict, Any, List, Optional
import os

# Import code generator for implementation suggestions
from backend.ml_analysis.code_generator import SimpleCodeGenerator

class BitAssistant:
    """
    Specialized AI assistant that analyzes machine learning models and provides
    actionable improvement suggestions with generated code examples.
    """
    
    def __init__(self, analysis_results=None, model_debugger=None):
        """
        Initialize the Bit assistant with analysis results or a model debugger.
        
        Args:
            analysis_results: Optional pre-computed analysis results
            model_debugger: Optional ModelDebugger instance
        """
        self.results = analysis_results
        self.debugger = model_debugger
        self.code_generator = SimpleCodeGenerator(api_key=os.environ.get("GEMINI_API_KEY"))
        self.framework = model_debugger.framework if model_debugger else "pytorch"
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("BitAssistant")
        
        # If we have a debugger but no results, run analysis
        if self.results is None and self.debugger is not None:
            self.logger.info("Running model analysis with debugger")
            self.results = self.debugger.analyze()
    
    def get_improvement_suggestions(self, detail_level="comprehensive") -> List[Dict[str, Any]]:
        """
        Get actionable suggestions to improve the model.
        
        Args:
            detail_level: Level of detail for suggestions ('basic', 'comprehensive', 'code')
            
        Returns:
            List of suggestion dictionaries
        """
        if self.debugger:
            # Use the existing improvement suggestion generator
            improvements = self.debugger.generate_improvement_suggestions(detail_level)
            return improvements.get("suggestions", [])
        
        elif self.results:
            # Generate suggestions based on the analysis results
            suggestions = []
            
            # Check accuracy level
            accuracy = self.results.get("accuracy", 0)
            if accuracy < 0.8:
                suggestions.append({
                    "category": "model_capacity",
                    "title": "Increase Model Complexity",
                    "issue": f"Model accuracy is only {accuracy*100:.1f}%, which suggests underfitting",
                    "suggestion": "Try increasing model complexity by adding more layers or parameters",
                    "severity": "high",
                    "expected_impact": "high"
                })
            
            # Check for other metrics and add suggestions
            if "confusion_matrix" in self.results:
                # Analyze confusion matrix for class-specific issues
                matrix = self.results["confusion_matrix"]
                if len(matrix.get("labels", [])) > 1:
                    suggestions.append({
                        "category": "evaluation",
                        "title": "Class-Specific Performance Issues",
                        "issue": "Some classes show significantly worse performance than others",
                        "suggestion": "Focus on improving performance for specific classes with higher error rates",
                        "severity": "medium",
                        "expected_impact": "medium-high"
                    })
            
            # Add general best practice suggestions
            suggestions.append({
                "category": "optimization",
                "title": "Optimize Hyperparameters",
                "issue": "Default hyperparameters may not be optimal",
                "suggestion": "Use grid search or Bayesian optimization to find better hyperparameters",
                "severity": "medium",
                "expected_impact": "medium"
            })
            
            return suggestions
            
        else:
            self.logger.error("No analysis results or debugger available")
            return []
    
    def generate_code_example(self, framework=None, category=None) -> str:
        """
        Generate implementation code for a specific suggestion category.
        
        Args:
            framework: Target ML framework (pytorch, tensorflow, sklearn)
            category: Type of suggestion to implement
            
        Returns:
            Generated code as string
        """
        if not framework:
            framework = self.framework
            
        if not category:
            # If no category specified, choose the most important one
            suggestions = self.get_improvement_suggestions()
            if suggestions:
                category = suggestions[0]["category"]
            else:
                category = "model_optimization"
        
        # Create context with model information
        model_context = {
            "accuracy": self.results.get("accuracy", 0) if self.results else 0,
            "framework": framework,
            "error_rate": self.results.get("error_analysis", {}).get("error_rate", 0) if self.results else 0
        }
        
        # Generate the code
        try:
            code = self.code_generator.generate_code_example(
                framework=framework,
                category=category,
                model_context=model_context
            )
            return code
        except Exception as e:
            self.logger.error(f"Error generating code: {str(e)}")
            return self._get_fallback_code(framework, category)
    
    def query(self, question: str) -> str:
        """
        Answer a natural language question about model optimization.
        
        Args:
            question: Natural language question about the model
            
        Returns:
            Response text with advice
        """
        # For now, implement a rule-based approach
        # In a production system, this would call an LLM API
        
        question_lower = question.lower()
        
        if "accuracy" in question_lower or "improve" in question_lower:
            accuracy = self.results.get("accuracy", 0) if self.results else 0
            
            if accuracy < 0.7:
                return (
                    f"Your model's accuracy is {accuracy*100:.1f}%, which is relatively low. "
                    "To improve accuracy, consider:\n"
                    "1. Increasing model complexity (more layers/parameters)\n"
                    "2. Using more training data or data augmentation\n"
                    "3. Optimizing learning rate and other hyperparameters\n"
                    "4. Adding regularization if you suspect overfitting on training data"
                )
            elif accuracy < 0.9:
                return (
                    f"Your model's accuracy is {accuracy*100:.1f}%, which is decent but can be improved. "
                    "Consider these targeted approaches:\n"
                    "1. Fine-tuning your learning rate schedule\n"
                    "2. Experimenting with different optimizers\n"
                    "3. Adding batch normalization for better training stability\n"
                    "4. Using ensemble methods to combine multiple models"
                )
            else:
                return (
                    f"Your model's accuracy is already quite high at {accuracy*100:.1f}%. "
                    "To squeeze out further improvements, try:\n"
                    "1. Ensemble methods (bagging, boosting)\n"
                    "2. Learning rate warm-up and cyclic learning rates\n"
                    "3. Fine-tuning with longer training and careful regularization\n"
                    "4. Advanced data augmentation techniques"
                )
        
        elif "overfitting" in question_lower or "regularization" in question_lower:
            return (
                "To address overfitting, try these techniques:\n"
                "1. Add dropout layers (start with 0.2-0.5 probability)\n"
                "2. Use L1/L2 regularization (weight decay in optimizers)\n"
                "3. Implement early stopping based on validation loss\n"
                "4. Try data augmentation to effectively increase training set size\n"
                "5. Reduce model complexity if the model is too large for your dataset"
            )
        
        # Default response for other questions
        return (
            "I can help optimize your model by analyzing its performance and structure. "
            "Try asking specific questions about accuracy, overfitting, hyperparameters, "
            "or architecture design. You can also use get_improvement_suggestions() to "
            "see a full list of recommended improvements."
        )
    
    def _get_fallback_code(self, framework, category):
        """Provide fallback code examples when API generation fails."""
        # Simplified version - in production would have more comprehensive examples
        if category == "class_imbalance":
            if framework == "pytorch":
                return """
# Handle class imbalance with weighted sampling
from torch.utils.data import WeightedRandomSampler
import torch

# Calculate class weights
y_train = torch.tensor(y_train)
class_counts = torch.bincount(y_train)
class_weights = 1.0 / class_counts.float()
weights = class_weights[y_train]

# Create weighted sampler
sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(y_train),
    replacement=True
)

# Use sampler in DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler  # Use the weighted sampler
)
"""
            elif framework == "tensorflow":
                return """
# Handle class imbalance with class weights
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# Use class weights in model.fit
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict  # Apply class weights
)
"""
            else:  # sklearn
                return """
# Handle class imbalance in scikit-learn
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Use SMOTE to oversample minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train model on balanced dataset
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)
"""
        
        # Generic optimization fallback
        return f"# {framework} optimization code for {category}\n# Implement your optimization strategy here"