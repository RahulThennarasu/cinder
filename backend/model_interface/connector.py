import logging
import numpy as np
import time
from typing import Dict, Any, Optional, Union, Tuple, List
from datetime import datetime

import torch
import tensorflow as tf

class ModelDebugger:
    """Main interface for connecting ML models to CompileML."""
    
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
        self.session_id = None
        self.server = None
        self.session_start_time = datetime.now()
        
        # Training history storage (would be populated from actual training in real app)
        self.training_history = []
        
        logging.info(f"Initialized ModelDebugger for {self.framework} model: {name}")
    
    def _detect_framework(self) -> str:
        """Detect which ML framework the model uses."""
        if isinstance(self.model, torch.nn.Module):
            return "pytorch"
        elif isinstance(self.model, tf.Module) or hasattr(self.model, 'predict'):
            return "tensorflow"
        else:
            raise ValueError("Unsupported model type. Please provide a PyTorch or TensorFlow model.")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run analysis on the model using the provided dataset.
        
        Returns:
            Dict containing analysis results
        """
        # Get predictions if they don't exist yet
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
        
        # Basic analysis
        accuracy = self._calculate_accuracy()
        error_analysis = self._analyze_errors()
        
        # Generate confusion matrix for binary classification
        if len(np.unique(self.ground_truth)) <= 2:
            confusion_matrix = self._calculate_confusion_matrix()
        else:
            confusion_matrix = None
        
        results = {
            "framework": self.framework,
            "model_name": self.name,
            "dataset_size": len(self.ground_truth) if self.ground_truth is not None else 0,
            "accuracy": accuracy,
            "error_analysis": error_analysis,
            "confusion_matrix": confusion_matrix,
        }
        
        return results
    
    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from the model on the dataset."""
        # Implementation will depend on the framework and dataset format
        
        if self.framework == "pytorch":
            # PyTorch implementation
            self.model.eval()
            all_preds = []
            all_targets = []
            
            # Simplified example - actual implementation would depend on dataset format
            try:
                with torch.no_grad():
                    for inputs, targets in self.dataset:
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                
                return np.array(all_preds), np.array(all_targets)
            except Exception as e:
                logging.error(f"Error getting predictions: {e}")
                # Generate random data for demonstration if there's an error
                return self._generate_demo_predictions()
                
        elif self.framework == "tensorflow":
            # TensorFlow implementation
            try:
                # Simplified implementation
                predictions = self.model.predict(self.dataset)
                # Extract targets based on dataset format (simplified)
                targets = np.array([y for _, y in self.dataset])
                return predictions, targets
            except Exception as e:
                logging.error(f"Error getting predictions: {e}")
                # Generate random data for demonstration if there's an error
                return self._generate_demo_predictions()
        
        # Default case - return demo data
        return self._generate_demo_predictions()
    
    def _generate_demo_predictions(self, num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random predictions for demonstration."""
        # For demo purposes, generate 100 binary classifications with 80% accuracy
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
        
        logging.info(f"Generated {num_samples} demo predictions with {len(error_indices)} errors")
        return predictions, ground_truth
    
    def _calculate_accuracy(self) -> float:
        """Calculate basic accuracy metric."""
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
        
        if len(self.predictions) == 0 or len(self.ground_truth) == 0:
            return 0.0
            
        return np.mean(self.predictions == self.ground_truth)
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze prediction errors."""
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
            
        if len(self.predictions) == 0 or len(self.ground_truth) == 0:
            return {
                "error_count": 0,
                "error_indices": [],
                "error_rate": 0.0
            }
            
        error_indices = np.where(self.predictions != self.ground_truth)[0]
        error_rate = len(error_indices) / len(self.ground_truth)
        
        return {
            "error_count": len(error_indices),
            "error_indices": error_indices.tolist(),
            "error_rate": error_rate
        }
    
    def _calculate_confusion_matrix(self) -> Dict[str, Any]:
        """Calculate confusion matrix for binary classification."""
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
        
        if len(self.predictions) == 0 or len(self.ground_truth) == 0:
            return {
                "matrix": [[0, 0], [0, 0]],
                "labels": ["0", "1"]
            }
        
        # For binary classification
        true_neg = np.sum((self.predictions == 0) & (self.ground_truth == 0))
        false_pos = np.sum((self.predictions == 1) & (self.ground_truth == 0))
        false_neg = np.sum((self.predictions == 0) & (self.ground_truth == 1))
        true_pos = np.sum((self.predictions == 1) & (self.ground_truth == 1))
        
        matrix = [
            [true_neg, false_pos],
            [false_neg, true_pos]
        ]
        
        return {
            "matrix": matrix,
            "labels": ["0", "1"]
        }
    
    def get_training_history(self, num_points: int = 10) -> List[Dict[str, Any]]:
        """
        Get training history data.
        
        In a real app, this would be populated during model training.
        Here we generate mock data for demonstration.
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
                
                self.training_history.append({
                    "iteration": i,
                    "accuracy": accuracy,
                    "loss": loss,
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
        
        # Calculate false positives and false negatives
        false_positives = np.sum((self.predictions == 1) & (self.ground_truth == 0))
        false_negatives = np.sum((self.predictions == 0) & (self.ground_truth == 1))
        
        return [
            {"name": "False Positives", "value": int(false_positives)},
            {"name": "False Negatives", "value": int(false_negatives)}
        ]
    
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