"""
ML Dashboard API - Main API Interface

This module provides a simple interface for connecting ML models to the dashboard.
"""

import os
import logging
import threading
from typing import Any, Dict, List, Optional, Union, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ml_dashboard")

class ModelDashboard:
    """
    Main interface for connecting ML models to the dashboard.
    
    Example:
        ```python
        from backend import ModelDashboard
        
        # Connect your model to the dashboard
        dashboard = ModelDashboard(model, dataset)
        
        # Launch the dashboard UI
        dashboard.launch(port=8000)
        ```
    """
    
    def __init__(self, 
                 model: Any, 
                 dataset: Any, 
                 name: str = "My Model", 
                 api_key: Optional[str] = None,
                 enable_code_generation: bool = True):
        """
        Initialize the dashboard with a model and dataset.
        
        Args:
            model: A PyTorch, TensorFlow, or scikit-learn model
            dataset: Dataset for evaluation (compatible with your model framework)
            name: Name for this model dashboard
            api_key: Optional Gemini API key for code generation features
            enable_code_generation: Whether to enable code generation features
        """
        self.model = model
        self.dataset = dataset
        self.name = name
        self.server = None
        self._debugger = None
        
        # Set API key if provided
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        
        # Import internal modules here to avoid circular imports
        try:
            from backend.model_interface.connector import ModelDebugger
            self._debugger = ModelDebugger(model, dataset, name)
            logger.info(f"Successfully initialized dashboard for {name}")
        except Exception as e:
            logger.error(f"Error initializing dashboard: {str(e)}")
            raise
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run comprehensive analysis on the model using the provided dataset.
        
        Returns:
            Dict containing analysis results including accuracy, precision, recall, etc.
        """
        if not self._debugger:
            raise ValueError("Dashboard not properly initialized")
        
        return self._debugger.analyze()
    
    def get_improvement_suggestions(self, detail_level: str = "comprehensive") -> Dict[str, Any]:
        """
        Get actionable suggestions to improve the model.
        
        Args:
            detail_level: Level of detail for suggestions ('basic', 'comprehensive', 'code')
        
        Returns:
            Dict with categorized improvement suggestions
        """
        if not self._debugger:
            raise ValueError("Dashboard not properly initialized")
        
        return self._debugger.get_improvement_suggestions(detail_level=detail_level)
    
    def launch(self, port: int = 8000, open_browser: bool = True) -> None:
        """
        Launch the dashboard server.
        
        Args:
            port: Port to run the dashboard on
            open_browser: Whether to automatically open the dashboard in a browser
        """
        if not self._debugger:
            raise ValueError("Dashboard not properly initialized")
        
        # Import the server module
        try:
            from backend.app.server import start_server
        except ImportError:
            raise ImportError("Could not import server module. Make sure the package is installed correctly.")
        
        # Open browser in a separate thread
        if open_browser:
            import webbrowser
            threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
        
        print(f"🔥 ML Dashboard is running at http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        
        # Start the server in the main thread
        start_server(self._debugger, port=port)
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """
        Analyze feature importance if the model supports it.
        
        Returns:
            Dict with feature importance information
        """
        if not self._debugger:
            raise ValueError("Dashboard not properly initialized")
        
        return self._debugger.analyze_feature_importance()
    
    def analyze_confidence(self) -> Dict[str, Any]:
        """
        Analyze the confidence of predictions.
        
        Returns:
            Dict with confidence metrics and distributions
        """
        if not self._debugger:
            raise ValueError("Dashboard not properly initialized")
        
        return self._debugger.analyze_confidence()
    
    def perform_cross_validation(self, k_folds: int = 5) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation on the model.
        
        Args:
            k_folds: Number of folds for cross-validation
        
        Returns:
            Dict with cross-validation results
        """
        if not self._debugger:
            raise ValueError("Dashboard not properly initialized")
        
        return self._debugger.perform_cross_validation(k_folds=k_folds)
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get training history data.
        
        Returns:
            List of training history entries
        """
        if not self._debugger:
            raise ValueError("Dashboard not properly initialized")
        
        return self._debugger.get_training_history()