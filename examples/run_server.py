#!/usr/bin/env python3
"""
run_server.py - Runs the CompileML API server with a real ML model
This file demonstrates how to integrate your ML model with the server.py API
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path

# Add the backend directory to the path if needed
current_dir = Path(__file__).parent
backend_dir = current_dir / "backend"
if backend_dir.exists():
    sys.path.insert(0, str(backend_dir))

# Import the server module
from server import start_server

# Example ML model implementations - choose based on your framework
def create_sklearn_model_debugger():
    """Create a debugger with a real scikit-learn model."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    
    class SklearnModelDebugger:
        def __init__(self):
            self.name = "Random Forest Classifier"
            self.framework = "sklearn"
            
            # Generate or load your dataset
            X, y = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=2,
                random_state=42
            )
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Train the model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Store test data and predictions
            self.X_test = X_test
            self.ground_truth = y_test
            self.predictions = self.model.predict(X_test)
            self.probabilities = self.model.predict_proba(X_test)
            
            # Store training data for additional analysis
            self.X_train = X_train
            self.y_train = y_train
            
            # Store the source file path for code retrieval
            self.source_file_path = __file__
        
        def analyze(self):
            """Perform comprehensive model analysis."""
            accuracy = accuracy_score(self.ground_truth, self.predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.ground_truth, self.predictions, average='weighted'
            )
            
            # ROC AUC for binary classification
            roc_auc = None
            roc_curve_data = None
            if len(np.unique(self.ground_truth)) == 2:
                roc_auc = roc_auc_score(self.ground_truth, self.probabilities[:, 1])
                from sklearn.metrics import roc_curve
                fpr, tpr, thresholds = roc_curve(self.ground_truth, self.probabilities[:, 1])
                roc_curve_data = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist()
                }
            
            # Error analysis
            error_indices = np.where(self.predictions != self.ground_truth)[0].tolist()
            error_count = len(error_indices)
            error_rate = error_count / len(self.ground_truth)
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(self.ground_truth, self.predictions)
            labels = [f"Class {i}" for i in range(len(np.unique(self.ground_truth)))]
            
            result = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": float(roc_auc) if roc_auc is not None else None,
                "error_analysis": {
                    "error_count": error_count,
                    "error_rate": float(error_rate),
                    "error_indices": error_indices,
                    "error_types": self._analyze_error_types()
                },
                "confusion_matrix": {
                    "matrix": cm.tolist(),
                    "labels": labels,
                    "num_classes": len(labels)
                }
            }
            
            if roc_curve_data:
                result["roc_curve"] = roc_curve_data
                
            return result
        
        def _analyze_error_types(self):
            """Analyze types of errors made by the model."""
            error_types = []
            unique_classes = np.unique(self.ground_truth)
            
            for true_class in unique_classes:
                for pred_class in unique_classes:
                    if true_class != pred_class:
                        count = np.sum(
                            (self.ground_truth == true_class) & 
                            (self.predictions == pred_class)
                        )
                        if count > 0:
                            error_types.append({
                                "name": f"Class {true_class} → Class {pred_class}",
                                "value": int(count),
                                "class_id": int(true_class)
                            })
            
            return error_types
        
        def analyze_confidence(self):
            """Analyze prediction confidence."""
            if self.probabilities is None:
                return {"error": "No probability scores available"}
            
            max_probs = np.max(self.probabilities, axis=1)
            correct_mask = self.predictions == self.ground_truth
            
            avg_confidence = np.mean(max_probs)
            avg_correct_confidence = np.mean(max_probs[correct_mask])
            avg_incorrect_confidence = np.mean(max_probs[~correct_mask])
            
            # Find overconfident and underconfident examples
            overconfident_mask = (max_probs > 0.8) & (~correct_mask)
            underconfident_mask = (max_probs < 0.6) & correct_mask
            
            return {
                "avg_confidence": float(avg_confidence),
                "avg_correct_confidence": float(avg_correct_confidence),
                "avg_incorrect_confidence": float(avg_incorrect_confidence),
                "calibration_error": float(abs(avg_confidence - np.mean(correct_mask))),
                "confidence_distribution": {
                    "bins": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "counts": np.histogram(max_probs, bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])[0].tolist()
                },
                "overconfident_examples": {
                    "indices": np.where(overconfident_mask)[0][:5].tolist(),
                    "confidences": max_probs[overconfident_mask][:5].tolist()
                },
                "underconfident_examples": {
                    "indices": np.where(underconfident_mask)[0][:5].tolist(),
                    "confidences": max_probs[underconfident_mask][:5].tolist()
                }
            }
        
        def analyze_feature_importance(self):
            """Analyze feature importance."""
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_names = [f"Feature_{i}" for i in range(len(importances))]
                
                return {
                    "feature_names": feature_names,
                    "importance_values": importances.tolist(),
                    "importance_method": "sklearn_feature_importances"
                }
            else:
                return {"error": "Feature importance not available for this model"}
        
        def generate_improvement_suggestions(self, detail_level="comprehensive"):
            """Generate improvement suggestions based on model analysis."""
            analysis = self.analyze()
            suggestions = []
            
            # Check accuracy
            if analysis["accuracy"] < 0.8:
                suggestions.append({
                    "category": "Model Performance",
                    "issue": f"Low accuracy: {analysis['accuracy']:.3f}",
                    "suggestion": "Consider trying different algorithms or hyperparameter tuning",
                    "severity": 0.8,
                    "impact": 0.9,
                    "code_example": """
# Try gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=200, max_depth=6)
model.fit(X_train, y_train)
"""
                })
            
            # Check error rate
            if analysis["error_analysis"]["error_rate"] > 0.2:
                suggestions.append({
                    "category": "Error Analysis",
                    "issue": f"High error rate: {analysis['error_analysis']['error_rate']:.3f}",
                    "suggestion": "Analyze misclassified samples and consider data preprocessing",
                    "severity": 0.7,
                    "impact": 0.8,
                    "code_example": """
# Feature scaling and selection
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
scaler = StandardScaler()
selector = SelectKBest(k=10)
X_scaled = scaler.fit_transform(X)
X_selected = selector.fit_transform(X_scaled, y)
"""
                })
            
            return {
                "suggestions": suggestions,
                "summary": {
                    "total_suggestions": len(suggestions),
                    "high_priority": sum(1 for s in suggestions if s["severity"] > 0.7),
                    "estimated_improvement": 0.1 * len(suggestions)
                }
            }
        
        def get_improvement_suggestions(self, detail_level="comprehensive"):
            """Alias for generate_improvement_suggestions."""
            return self.generate_improvement_suggestions(detail_level)
        
        def perform_cross_validation(self, k_folds=5):
            """Perform cross-validation."""
            from sklearn.model_selection import cross_validate
            
            # Combine train and test data for CV
            X_all = np.vstack([self.X_train, self.X_test])
            y_all = np.hstack([self.y_train, self.ground_truth])
            
            cv_results = cross_validate(
                self.model, X_all, y_all, cv=k_folds,
                scoring=['accuracy', 'precision_weighted', 'recall_weighted'],
                return_train_score=False
            )
            
            fold_results = []
            for i in range(k_folds):
                fold_results.append({
                    "fold": i + 1,
                    "accuracy": float(cv_results['test_accuracy'][i]),
                    "precision": float(cv_results['test_precision_weighted'][i]),
                    "recall": float(cv_results['test_recall_weighted'][i])
                })
            
            return {
                "fold_results": fold_results,
                "mean_accuracy": float(np.mean(cv_results['test_accuracy'])),
                "std_accuracy": float(np.std(cv_results['test_accuracy'])),
                "n_folds": k_folds
            }
        
        def analyze_prediction_drift(self, threshold=0.1):
            """Analyze prediction drift."""
            true_dist = {str(cls): int(np.sum(self.ground_truth == cls)) 
                        for cls in np.unique(self.ground_truth)}
            pred_dist = {str(cls): int(np.sum(self.predictions == cls)) 
                        for cls in np.unique(self.predictions)}
            
            # Calculate drift scores (simplified)
            drift_scores = {}
            drifting_classes = []
            
            for cls in true_dist.keys():
                true_ratio = true_dist[cls] / len(self.ground_truth)
                pred_ratio = pred_dist.get(cls, 0) / len(self.predictions)
                drift_score = abs(true_ratio - pred_ratio)
                drift_scores[cls] = drift_score
                
                if drift_score > threshold:
                    drifting_classes.append(int(cls))
            
            overall_drift = np.mean(list(drift_scores.values()))
            
            return {
                "class_distribution": true_dist,
                "prediction_distribution": pred_dist,
                "drift_scores": drift_scores,
                "drifting_classes": drifting_classes,
                "overall_drift": float(overall_drift)
            }
        
        def get_training_history(self):
            """Return mock training history (sklearn doesn't track this by default)."""
            return [
                {
                    "iteration": i + 1,
                    "accuracy": 0.5 + 0.3 * (1 - np.exp(-i/20)),
                    "loss": 2.0 * np.exp(-i/20),
                    "learning_rate": 0.001,
                    "timestamp": f"2024-01-01T{10+i//6:02d}:{(i*10)%60:02d}:00"
                }
                for i in range(50)
            ]
        
        def analyze_error_types(self):
            """Return error type analysis."""
            return self._analyze_error_types()
        
        def get_sample_predictions(self, limit=10, offset=0, include_errors_only=False):
            """Get sample predictions with details."""
            if include_errors_only:
                error_mask = self.predictions != self.ground_truth
                indices = np.where(error_mask)[0][offset:offset+limit]
            else:
                indices = range(offset, min(offset+limit, len(self.predictions)))
            
            samples = []
            for idx in indices:
                if idx < len(self.predictions):
                    samples.append({
                        "index": int(idx),
                        "prediction": int(self.predictions[idx]),
                        "true_label": int(self.ground_truth[idx]),
                        "is_error": bool(self.predictions[idx] != self.ground_truth[idx]),
                        "confidence": float(np.max(self.probabilities[idx])) if self.probabilities is not None else None,
                        "probabilities": self.probabilities[idx].tolist() if self.probabilities is not None else None
                    })
            
            return {
                "samples": samples,
                "total": len(self.predictions),
                "limit": limit,
                "offset": offset,
                "include_errors_only": include_errors_only
            }
    
    return SklearnModelDebugger()


def create_pytorch_model_debugger():
    """Create a debugger with a real PyTorch model."""
    # You can implement this for PyTorch models
    # Similar structure to sklearn but with PyTorch-specific code
    pass


def create_tensorflow_model_debugger():
    """Create a debugger with a real TensorFlow model."""
    # You can implement this for TensorFlow models
    # Similar structure to sklearn but with TensorFlow-specific code
    pass


def main():
    """Main function to run the server with a real model."""
    parser = argparse.ArgumentParser(description='Run CompileML API Server with ML Model')
    parser.add_argument('--port', type=int, default=8000, 
                       help='Port to run the server on (default: 8000)')
    parser.add_argument('--framework', choices=['sklearn', 'pytorch', 'tensorflow'], 
                       default='sklearn', help='ML framework to use (default: sklearn)')
    parser.add_argument('--model-path', type=str, 
                       help='Path to saved model file (optional)')
    parser.add_argument('--data-path', type=str,
                       help='Path to dataset file (optional)')
    
    args = parser.parse_args()
    
    print(f"Starting CompileML API Server with {args.framework} model...")
    print(f"Server will be available at: http://localhost:{args.port}")
    print("API documentation: http://localhost:{args.port}/docs")
    print("Dashboard UI: http://localhost:{args.port}")
    
    # Create the appropriate model debugger
    if args.framework == 'sklearn':
        model_debugger = create_sklearn_model_debugger()
    elif args.framework == 'pytorch':
        model_debugger = create_pytorch_model_debugger()
        if model_debugger is None:
            print("PyTorch debugger not implemented yet. Using sklearn instead.")
            model_debugger = create_sklearn_model_debugger()
    elif args.framework == 'tensorflow':
        model_debugger = create_tensorflow_model_debugger()
        if model_debugger is None:
            print("TensorFlow debugger not implemented yet. Using sklearn instead.")
            model_debugger = create_sklearn_model_debugger()
    else:
        raise ValueError(f"Unknown framework: {args.framework}")
    
    print(f"Model loaded: {model_debugger.name}")
    print(f"Framework: {model_debugger.framework}")
    print(f"Dataset size: {len(model_debugger.ground_truth)} samples")
    
    # Start the server with the model debugger
    start_server(model_debugger, port=args.port)


if __name__ == "__main__":
    main()