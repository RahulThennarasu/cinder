import logging
from typing import Dict, Any, Optional, Union, Tuple

import numpy as np
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
        # Get predictions
        self.predictions, self.ground_truth = self._get_predictions()
        
        # Basic analysis
        results = {
            "framework": self.framework,
            "model_name": self.name,
            "dataset_size": len(self.ground_truth),
            "accuracy": self._calculate_accuracy(),
            "error_analysis": self._analyze_errors(),
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
                # Return empty arrays instead of None
                return np.array([]), np.array([])
                
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
                return np.array([]), np.array([])
        
        # Default case - return empty arrays
        return np.array([]), np.array([])
    
    def _calculate_accuracy(self) -> float:
        """Calculate basic accuracy metric."""
        if self.predictions is None:
            self.analyze()
        return np.mean(self.predictions == self.ground_truth)
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze prediction errors."""
        if self.predictions is None:
            self.analyze()
            
        error_indices = np.where(self.predictions != self.ground_truth)[0]
        
        return {
            "error_count": len(error_indices),
            "error_indices": error_indices.tolist()[:100],  # Limit to first 100
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