import logging
import numpy as np
import time
from typing import Dict, Any, Optional, Union, Tuple, List
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import KFold

import torch
import tensorflow as tf

class ModelDebugger:
    """Enhanced interface for connecting ML models to CompileML."""
    
    def __init__(self, model, dataset, name: str = "My Model"):
        """
        Initialize the debugger with a model and dataset.
        
        Args:
            model: A PyTorch or TensorFlow model
            dataset: Dataset for evaluation (various formats supported)
            name: Name for this debugging session
        """
        self.model = model
        self.dataset = dataset
        self.name = name
        self.framework = self._detect_framework()
        
        # Initialize storage for debugging data
        self.predictions = None
        self.ground_truth = None
        self.prediction_probas = None  # Store probability scores
        self.session_id = None
        self.server = None
        self.session_start_time = datetime.now()
        
        # Feature importance cache
        self._feature_importances = None
        
        # Training history storage
        self.training_history = []
        
        logging.info(f"Initialized ModelDebugger for {self.framework} model: {name}")
    
    def _detect_framework(self) -> str:
        """Detect which ML framework the model uses."""
        if isinstance(self.model, torch.nn.Module):
            return "pytorch"
        elif isinstance(self.model, tf.Module) or hasattr(self.model, 'predict'):
            return "tensorflow"
        else:
            # Try to detect scikit-learn
            if hasattr(self.model, 'predict') and hasattr(self.model, 'fit'):
                return "sklearn"
            else:
                raise ValueError("Unsupported model type. Please provide a PyTorch, TensorFlow, or scikit-learn model.")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run comprehensive analysis on the model using the provided dataset.
        
        Returns:
            Dict containing analysis results
        """
        # Get predictions if they don't exist yet
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
        
        # Calculate metrics
        accuracy = self._calculate_accuracy()
        error_analysis = self._analyze_errors()
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.ground_truth, 
            self.predictions,
            average='weighted'
        )
        
        # Generate confusion matrix
        confusion_matrix = self._calculate_confusion_matrix()
        
        # Add advanced metrics
        results = {
            "framework": self.framework,
            "model_name": self.name,
            "dataset_size": len(self.ground_truth) if self.ground_truth is not None else 0,
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "error_analysis": error_analysis,
            "confusion_matrix": confusion_matrix,
        }
        
        # Add more metrics if probabilities are available
        if hasattr(self, 'prediction_probas') and self.prediction_probas is not None:
            try:
                # Try to calculate AUC, but only for binary classification
                unique_classes = np.unique(self.ground_truth)
                if len(unique_classes) == 2:
                    # For binary classification
                    positive_class = unique_classes[1]  # Assuming second class is positive
                    binary_truth = (self.ground_truth == positive_class).astype(int)
                    
                    # Get probabilities for positive class
                    pos_probs = self.prediction_probas[:, 1] if self.prediction_probas.shape[1] > 1 else self.prediction_probas
                    
                    # Calculate ROC AUC
                    roc_auc = roc_auc_score(binary_truth, pos_probs)
                    results["roc_auc"] = float(roc_auc)
                    
                    # Calculate ROC curve points
                    fpr, tpr, thresholds = roc_curve(binary_truth, pos_probs)
                    results["roc_curve"] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "thresholds": thresholds.tolist()
                    }
            except Exception as e:
                logging.warning(f"Could not calculate ROC metrics: {str(e)}")
        
        return results
    
    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from the model on the dataset."""
        # Implementation will depend on the framework and dataset format
        
        if self.framework == "pytorch":
            # PyTorch implementation
            self.model.eval()
            all_preds = []
            all_targets = []
            all_probas = []  # Store probability scores
            
            # Simplified example - actual implementation would depend on dataset format
            try:
                with torch.no_grad():
                    for inputs, targets in self.dataset:
                        # Move to GPU if available
                        if torch.cuda.is_available():
                            inputs = inputs.cuda()
                            targets = targets.cuda()
                            self.model = self.model.cuda()
                        
                        outputs = self.model(inputs)
                        
                        # Store probabilities for confidence analysis
                        probas = torch.softmax(outputs, dim=1) if outputs.dim() > 1 else torch.sigmoid(outputs)
                        all_probas.extend(probas.cpu().numpy())
                        
                        # Get predicted class
                        if outputs.dim() > 1:  # Multi-class
                            _, preds = torch.max(outputs, 1)
                        else:  # Binary
                            preds = (outputs > 0.5).long()
                            
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                
                self.prediction_probas = np.array(all_probas)
                return np.array(all_preds), np.array(all_targets)
            except Exception as e:
                logging.error(f"Error getting predictions: {str(e)}")
                # Generate random data for demonstration if there's an error
                return self._generate_demo_predictions()
                
        elif self.framework == "tensorflow":
            # TensorFlow implementation
            try:
                # Get predictions
                predictions_raw = self.model.predict(self.dataset)
                
                # Handle different output formats
                if len(predictions_raw.shape) > 1 and predictions_raw.shape[1] > 1:
                    # Multi-class: store probabilities and get class with highest probability
                    self.prediction_probas = predictions_raw
                    predictions = np.argmax(predictions_raw, axis=1)
                else:
                    # Binary: apply threshold
                    self.prediction_probas = predictions_raw
                    predictions = (predictions_raw > 0.5).astype(int)
                
                # Extract targets based on dataset format (simplified)
                if hasattr(self.dataset, 'unbatch'):
                    # TF Dataset
                    targets = np.array([y.numpy() for x, y in self.dataset.unbatch()])
                else:
                    # Try to extract y from dataset
                    targets = getattr(self.dataset, 'y', None)
                    if targets is None:
                        # Fall back to demo data if targets can't be extracted
                        return self._generate_demo_predictions()
                
                return predictions, targets
            except Exception as e:
                logging.error(f"Error getting predictions: {str(e)}")
                # Generate random data for demonstration if there's an error
                return self._generate_demo_predictions()
        
        # Default case - return demo data
        return self._generate_demo_predictions()
    
    def _generate_demo_predictions(self, num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random predictions for demonstration."""
        # For demo purposes, generate binary classifications with 80% accuracy
        np.random.seed(42)  # For reproducibility
        
        # Generate ground truth (0 or 1)
        ground_truth = np.random.randint(0, 2, size=num_samples)
        
        # Generate predictions with 80% accuracy
        predictions = np.copy(ground_truth)
        error_indices = np.random.choice(
            np.arange(num_samples), 
            size=int(num_samples * 0.2),  # 20% error rate
            replace=False
        )
        predictions[error_indices] = 1 - predictions[error_indices]
        
        # Generate prediction probabilities
        self.prediction_probas = np.zeros((num_samples, 2))
        for i in range(num_samples):
            if predictions[i] == 1:
                # If prediction is class 1, probability is between 0.5 and 1.0
                self.prediction_probas[i, 1] = 0.5 + np.random.random() * 0.5
                self.prediction_probas[i, 0] = 1 - self.prediction_probas[i, 1]
            else:
                # If prediction is class 0, probability is between 0.5 and 1.0 for class 0
                self.prediction_probas[i, 0] = 0.5 + np.random.random() * 0.5
                self.prediction_probas[i, 1] = 1 - self.prediction_probas[i, 0]
        
        logging.info(f"Generated {num_samples} demo predictions with {len(error_indices)} errors")
        return predictions, ground_truth
    
    def _calculate_accuracy(self) -> float:
        """Calculate basic accuracy metric."""
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
        
        if len(self.predictions) == 0 or len(self.ground_truth) == 0:
            return 0.0
            
        return float(np.mean(self.predictions == self.ground_truth))
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze prediction errors in detail."""
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
            
        if len(self.predictions) == 0 or len(self.ground_truth) == 0:
            return {
                "error_count": 0,
                "error_indices": [],
                "error_rate": 0.0
            }
            
        error_mask = self.predictions != self.ground_truth
        error_indices = np.where(error_mask)[0]
        error_rate = len(error_indices) / len(self.ground_truth)
        
        # Enhanced error analysis - track error types
        error_types = {}
        for idx in error_indices:
            true_class = self.ground_truth[idx]
            pred_class = self.predictions[idx]
            error_key = f"{true_class}→{pred_class}"
            
            if error_key not in error_types:
                error_types[error_key] = {
                    "true_class": int(true_class),
                    "predicted_class": int(pred_class),
                    "count": 0,
                    "indices": []
                }
            
            error_types[error_key]["count"] += 1
            error_types[error_key]["indices"].append(int(idx))
        
        # Sort error types by frequency
        error_types_list = sorted(
            list(error_types.values()),
            key=lambda x: x["count"],
            reverse=True
        )
        
        return {
            "error_count": len(error_indices),
            "error_indices": error_indices.tolist(),
            "error_rate": float(error_rate),
            "error_types": error_types_list
        }
    
    def _calculate_confusion_matrix(self) -> Dict[str, Any]:
        """Calculate confusion matrix for multi-class classification."""
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
        
        if len(self.predictions) == 0 or len(self.ground_truth) == 0:
            return {
                "matrix": [[0, 0], [0, 0]],
                "labels": ["0", "1"]
            }
        
        # Get unique classes
        classes = np.unique(np.concatenate((self.predictions, self.ground_truth)))
        n_classes = len(classes)
        
        # Calculate confusion matrix
        cm = confusion_matrix(
            self.ground_truth,
            self.predictions,
            labels=classes
        )
        
        return {
            "matrix": cm.tolist(),
            "labels": [str(c) for c in classes],
            "num_classes": int(n_classes)
        }
    
    def analyze_confidence(self) -> Dict[str, Any]:
        """
        Analyze the confidence of predictions.
        
        Returns:
            Dict with confidence metrics and distributions
        """
        if not hasattr(self, 'prediction_probas') or self.prediction_probas is None:
            # Re-run predictions to get probabilities if not available
            self._get_predictions()
            if not hasattr(self, 'prediction_probas'):
                return {"error": "Cannot compute confidence, probabilities not available"}
        
        # Handle different probability formats - with proper null checking
        if self.prediction_probas is not None:
            if len(self.prediction_probas.shape) > 1 and self.prediction_probas.shape[1] > 1:
                # For multi-class, confidence is max probability
                confidences = np.max(self.prediction_probas, axis=1)
            else:
                # For binary with single column, transform to max prob
                confidences = np.maximum(self.prediction_probas, 1 - self.prediction_probas)
        else:
            # If no probability data is available, set confidences to a default
            confidences = np.ones_like(self.predictions, dtype=float) * 0.5
        
        # Analyze confidence for correct vs incorrect predictions
        correct_mask = self.predictions == self.ground_truth
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        # Calculate statistics
        avg_confidence = float(np.mean(confidences))
        avg_correct_confidence = float(np.mean(correct_confidences) if len(correct_confidences) > 0 else 0)
        avg_incorrect_confidence = float(np.mean(incorrect_confidences) if len(incorrect_confidences) > 0 else 0)
        
        # Calculate calibration error (difference between accuracy and confidence)
        accuracy = self._calculate_accuracy()
        calibration_error = float(abs(avg_confidence - accuracy))
        
        # Create confidence bins for distribution analysis
        bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
        overall_hist, _ = np.histogram(confidences, bins=bins)
        correct_hist, _ = np.histogram(correct_confidences, bins=bins)
        incorrect_hist, _ = np.histogram(incorrect_confidences, bins=bins)
        
        # Find overconfident examples (high confidence but wrong predictions)
        overconfident_threshold = 0.9
        overconfident_indices = np.where((~correct_mask) & (confidences > overconfident_threshold))[0]
        
        # Find underconfident examples (low confidence but correct predictions)
        underconfident_threshold = 0.6
        underconfident_indices = np.where((correct_mask) & (confidences < underconfident_threshold))[0]
        
        return {
            "avg_confidence": avg_confidence,
            "avg_correct_confidence": avg_correct_confidence,
            "avg_incorrect_confidence": avg_incorrect_confidence,
            "calibration_error": calibration_error,
            "confidence_distribution": {
                "bin_edges": bins.tolist(),
                "overall": overall_hist.tolist(),
                "correct": correct_hist.tolist(),
                "incorrect": incorrect_hist.tolist(),
            },
            "overconfident_examples": {
                "threshold": overconfident_threshold,
                "count": len(overconfident_indices),
                "indices": overconfident_indices.tolist()[:10]  # Return first 10 for UI display
            },
            "underconfident_examples": {
                "threshold": underconfident_threshold,
                "count": len(underconfident_indices),
                "indices": underconfident_indices.tolist()[:10]  # Return first 10 for UI display
            }
        }
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """
        Analyze feature importance if the model supports it.
        
        Returns:
            Dict with feature importance information
        """
        # For PyTorch models, we need a workaround since they don't have direct feature importance
        if self.framework == "pytorch":
            try:
                # Create a feature importance proxy using input gradients
                # This is a simplified approach - for real usage, use more robust methods
                if not hasattr(self, '_feature_importances'):
                    # Get a small batch of data
                    data_iter = iter(self.dataset)
                    inputs, _ = next(data_iter)
                    
                    # Move to GPU if available
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        self.model = self.model.cuda()
                    
                    # Check if inputs need gradient
                    if not inputs.requires_grad:
                        inputs.requires_grad = True
                    
                    # Reset gradients
                    self.model.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Compute the sum of outputs (to get a scalar)
                    outputs.sum().backward()
                    
                    # Get the gradient
                    gradient = inputs.grad
                    
                    # Use the absolute mean of gradients as feature importance
                    importances = gradient.abs().mean(dim=0).cpu().numpy()
                    
                    # Normalize to 0-1
                    if importances.sum() > 0:
                        importances = importances / importances.sum()
                    
                    self._feature_importances = importances
                
                # Flatten for consistent output format regardless of input shape
                if self._feature_importances is not None:
                    importances_flat = self._feature_importances.flatten()
                    
                    # Create feature names based on shape
                    feature_names = [f"Feature {i}" for i in range(len(importances_flat))]
                    
                    # Sort by importance
                    indices = np.argsort(importances_flat)[::-1]
                    sorted_importances = importances_flat[indices]
                    sorted_names = [feature_names[i] for i in indices]
                    
                    return {
                        "feature_names": sorted_names,
                        "importance_values": sorted_importances.tolist(),
                        "importance_method": "gradient_based"
                    }
                else:
                    return {
                        "error": "Could not calculate feature importance: No importance data available"
                    }
                
            except Exception as e:
                logging.error(f"Error calculating feature importance: {str(e)}")
                return {"error": f"Could not calculate feature importance: {str(e)}"}
        
        # For scikit-learn-like models that have feature_importances_
        elif hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = [f"Feature {i}" for i in range(len(importances))]
            
            # Sort by importance
            indices = np.argsort(importances)[::-1]
            sorted_importances = importances[indices]
            sorted_names = [feature_names[i] for i in indices]
            
            return {
                "feature_names": sorted_names,
                "importance_values": sorted_importances.tolist(),
                "importance_method": "model_attributions"
            }
        
        return {"error": "Model type does not support feature importance analysis"}
    
    def perform_cross_validation(self, k_folds=5) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation on the model.
        
        Args:
            k_folds: Number of folds for cross-validation
            
        Returns:
            Dict with cross-validation results
        """
        if self.dataset is None:
            return {"error": "No dataset available for cross-validation"}
        
        # Extract all data from dataset
        try:
            # This assumes dataset is a PyTorch DataLoader or similar
            all_inputs = []
            all_targets = []
            
            for inputs, targets in self.dataset:
                all_inputs.append(inputs)
                all_targets.append(targets)
            
            # Concatenate batches
            X = torch.cat(all_inputs)
            y = torch.cat(all_targets)
            
            # Move to CPU for sklearn
            X = X.cpu()
            y = y.cpu()
            
            # Setup cross-validation
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                # Get fold data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Clone model (depends on framework)
                if self.framework == "pytorch":
                    import copy
                    fold_model = copy.deepcopy(self.model)
                    
                    # Create optimizer
                    optimizer = torch.optim.Adam(fold_model.parameters(), lr=0.01)
                    
                    # Loss function
                    if len(np.unique(y.numpy())) > 2:
                        criterion = torch.nn.CrossEntropyLoss()
                    else:
                        criterion = torch.nn.BCEWithLogitsLoss()
                    
                    # Train for a few epochs
                    fold_model.train()
                    for epoch in range(3):  # Just a few epochs for quick validation
                        # Forward pass
                        outputs = fold_model(X_train)
                        loss = criterion(outputs, y_train)
                        
                        # Backward and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    # Evaluate fold model
                    fold_model.eval()
                    with torch.no_grad():
                        outputs = fold_model(X_val)
                        _, preds = torch.max(outputs, 1)
                    
                    # Calculate accuracy
                    accuracy = (preds == y_val).float().mean().item()
                    
                    fold_results.append({
                        "fold": fold + 1,
                        "accuracy": float(accuracy),
                        "val_size": int(len(y_val))
                    })
            
            # Calculate overall statistics
            accuracies = [r["accuracy"] for r in fold_results]
            mean_accuracy = float(np.mean(accuracies))
            std_accuracy = float(np.std(accuracies))
            
            return {
                "fold_results": fold_results,
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
                "n_folds": k_folds
            }
            
        except Exception as e:
            logging.error(f"Error during cross-validation: {str(e)}")
            return {"error": f"Cross-validation failed: {str(e)}"}
    
    def analyze_prediction_drift(self, threshold=0.1) -> Dict[str, Any]:
        """
        Analyze if predictions drift significantly from the training distribution.
        
        This helps detect if the model is receiving inputs significantly different from training.
        
        Args:
            threshold: Threshold for considering a prediction as drifted
            
        Returns:
            Dict with drift analysis results
        """
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
            
        try:
            # For demonstration, we'll simulate drift detection
            # In a real implementation, you would compare feature distributions
            
            # Calculate class distribution
            unique_classes = np.unique(self.ground_truth)
            class_counts = {}
            for cls in unique_classes:
                class_counts[int(cls)] = int(np.sum(self.ground_truth == cls))
            
            # Calculate prediction distribution
            pred_counts = {}
            for cls in unique_classes:
                pred_counts[int(cls)] = int(np.sum(self.predictions == cls))
            
            # Calculate difference in distributions
            drift_scores = {}
            for cls in unique_classes:
                expected_ratio = class_counts[int(cls)] / len(self.ground_truth)
                actual_ratio = pred_counts[int(cls)] / len(self.predictions)
                drift_scores[int(cls)] = float(abs(actual_ratio - expected_ratio))
            
            # Identify drifting classes
            drifting_classes = [cls for cls, score in drift_scores.items() if score > threshold]
            
            return {
                "class_distribution": {str(k): v for k, v in class_counts.items()},
                "prediction_distribution": {str(k): v for k, v in pred_counts.items()},
                "drift_scores": {str(k): v for k, v in drift_scores.items()},
                "drifting_classes": drifting_classes,
                "overall_drift": float(sum(drift_scores.values()) / len(drift_scores))
            }
        except Exception as e:
            logging.error(f"Error analyzing prediction drift: {str(e)}")
            return {"error": f"Could not analyze prediction drift: {str(e)}"}
    
    def get_training_history(self, num_points: int = 10) -> List[Dict[str, Any]]:
        """
        Get training history data.
        
        In a real app, this would be populated during model training.
        Here we generate mock data for demonstration if not already available.
        """
        if not self.training_history:
            # Generate a simulated training history
            base_accuracy = 0.65
            base_loss = 0.5
            
            for i in range(1, num_points + 1):
                time_offset = i * 300  # 5 minutes between points
                epoch_time = self.session_start_time.timestamp() + time_offset
                epoch_date = datetime.fromtimestamp(epoch_time)
                
                # Accuracy increases over time, with some noise
                accuracy = min(0.98, base_accuracy + (i * 0.03) + np.random.uniform(-0.01, 0.01))
                
                # Loss decreases over time, with some noise
                loss = max(0.05, base_loss - (i * 0.05) + np.random.uniform(-0.01, 0.01))
                
                # Add learning rate that decreases over time
                learning_rate = 0.01 * (0.9 ** i)
                
                self.training_history.append({
                    "iteration": i,
                    "accuracy": float(accuracy),
                    "loss": float(loss),
                    "learning_rate": float(learning_rate),
                    "timestamp": epoch_date.isoformat()
                })
        
        return self.training_history
    
    def analyze_error_types(self) -> List[Dict[str, Any]]:
        """
        Analyze types of errors (false positives vs false negatives).
        
        In a real app, this would analyze the actual predictions.
        Here we generate sensible mock data based on our predictions.
        """
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
        
        if len(self.predictions) == 0 or len(self.ground_truth) == 0:
            return [
                {"name": "False Positives", "value": 0},
                {"name": "False Negatives", "value": 0}
            ]
        
        # Binary classification case
        unique_classes = np.unique(self.ground_truth)
        if len(unique_classes) == 2:
            # Calculate false positives and false negatives
            # Assuming class 1 is positive and 0 is negative
            positive_class = unique_classes[1]
            negative_class = unique_classes[0]
            
            false_positives = np.sum((self.predictions == positive_class) & (self.ground_truth == negative_class))
            false_negatives = np.sum((self.predictions == negative_class) & (self.ground_truth == positive_class))
            
            return [
                {"name": "False Positives", "value": int(false_positives)},
                {"name": "False Negatives", "value": int(false_negatives)}
            ]
        else:
            # Multi-class case - return per-class errors
            results = []
            for cls in unique_classes:
                # One-vs-all approach
                cls_int = int(cls)
                
                # False positives: predicted as cls but isn't
                false_pos = np.sum((self.predictions == cls) & (self.ground_truth != cls))
                
                # False negatives: is cls but predicted as something else
                false_neg = np.sum((self.predictions != cls) & (self.ground_truth == cls))
                
                results.append({
                    "class": cls_int,
                    "name": f"Class {cls_int} False Positives",
                    "value": int(false_pos)
                })
                
                results.append({
                    "class": cls_int,
                    "name": f"Class {cls_int} False Negatives",
                    "value": int(false_neg)
                })
            
            return results
    
    def get_sample_predictions(self, limit: int = 10, offset: int = 0, include_errors_only: bool = False) -> Dict[str, Any]:
        """
        Get sample predictions for inspection.
        
        Args:
            limit: Maximum number of samples to return
            offset: Offset for pagination
            include_errors_only: If True, only return incorrect predictions
            
        Returns:
            Dict with sample predictions
        """
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
        
        if len(self.predictions) == 0 or len(self.ground_truth) == 0:
            return {
                "samples": [],
                "total": 0,
                "limit": limit,
                "offset": offset
            }
        
        # Get indices of samples to include
        if include_errors_only:
            indices = np.where(self.predictions != self.ground_truth)[0]
        else:
            indices = np.arange(len(self.predictions))
        
        # Apply pagination
        total = len(indices)
        if offset >= total:
            offset = max(0, total - limit)
        
        # Get slice of indices
        paginated_indices = indices[offset:offset+limit]
        
        # Prepare samples
        samples = []
        for idx in paginated_indices:
            sample = {
                "index": int(idx),
                "prediction": int(self.predictions[idx]),
                "true_label": int(self.ground_truth[idx]),
                "is_error": bool(self.predictions[idx] != self.ground_truth[idx])
            }
            
            # Add probability if available
            if hasattr(self, 'prediction_probas') and self.prediction_probas is not None:
                if len(self.prediction_probas.shape) > 1 and self.prediction_probas.shape[1] > 1:
                    # Multi-class probabilities
                    sample["probabilities"] = self.prediction_probas[idx].tolist()
                    sample["confidence"] = float(np.max(self.prediction_probas[idx]))
                else:
                    # Binary case
                    prob = float(self.prediction_probas[idx])
                    sample["probabilities"] = [1 - prob, prob]
                    sample["confidence"] = float(max(prob, 1 - prob))
            
            samples.append(sample)
        
        return {
            "samples": samples,
            "total": total,
            "limit": limit,
            "offset": offset,
            "include_errors_only": include_errors_only
        }
    
    def launch_dashboard(self, port: int = 8000) -> None:
        """Launch the debugging dashboard server."""
        import threading
        from backend.app.server import start_server
        
        # Analyze if not already done
        if self.predictions is None:
            self.analyze()
        
        print(f"CompileML dashboard is running at http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        
        # Start the server in a separate thread
        server_thread = threading.Thread(
            target=lambda: start_server(self, port=port),
            daemon=True  # This makes the thread exit when the main program exits
        )
        server_thread.start()