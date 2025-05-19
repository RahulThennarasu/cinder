import numpy as np
import matplotlib.pyplot as plt
import os

class PredictionExplainer:
    def __init__(self, model_debugger):
        self.model_debugger = model_debugger
        self.model = model_debugger.model
        
    def visualize_prediction_explanation(self, instance_idx, method='shap', save_path=None):
        """Generate a visualization explaining a specific prediction"""
        # Get the instance
        X = np.array([x[0].numpy() for x, _ in self.model_debugger.dataset])
        instance = X[instance_idx]
        
        # Get feature names (if available)
        if hasattr(self.model_debugger, 'feature_names'):
            feature_names = self.model_debugger.feature_names
        else:
            feature_names = [f"Feature {i}" for i in range(instance.shape[0])]
        
        # Method 1: SHAP values (requires shap package)
        if method == 'shap':
            import shap
            
            # Create an explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                # For sklearn models
                explainer = shap.Explainer(self.model)
                shap_values = explainer(instance.reshape(1, -1))
                
                plt.figure(figsize=(12, 6))
                if isinstance(shap_values, list):  # For multi-class models
                    prediction_class = self.model_debugger.predictions[instance_idx]
                    base_value = shap_values[prediction_class].base_values
                    shap.plots.force(
                        base_value, 
                        shap_values[prediction_class].values[0],
                        features=instance,
                        feature_names=feature_names,
                        matplotlib=True
                    )
                else:  # For binary models
                    shap.plots.force(
                        shap_values.base_values[0], 
                        shap_values[0],
                        features=instance,
                        feature_names=feature_names,
                        matplotlib=True
                    )
            
            # Custom implementation for PyTorch models would be needed here
            
        # Method 2: Simple feature contribution visualization
        elif method == 'simple':
            # Get importance scores (this is a simple approach)
            importance_data = self.model_debugger.analyze_feature_importance()
            importance_values = np.array(importance_data["importance_values"])
            
            # Scale the instance values by importance
            contributions = instance * importance_values
            
            # Plot contributions
            plt.figure(figsize=(12, 6))
            plt.bar(feature_names, contributions)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Feature Contributions for Instance {instance_idx}')
            plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            plt.show()