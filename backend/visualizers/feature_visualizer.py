import numpy as np
import matplotlib.pyplot as plt
import os

class FeatureVisualizer:
    def __init__(self, model_debugger):
        self.model_debugger = model_debugger
        
    def generate_feature_importance_plot(self, save_path=None):
        """Generate a horizontal bar chart of feature importances"""
        importance_data = self.model_debugger.analyze_feature_importance()
        
        # Sort features by importance
        feature_names = importance_data["feature_names"]
        importance_values = importance_data["importance_values"]
        
        # Create sorting indices
        sorted_indices = np.argsort(importance_values)
        
        # Plot horizontal bar chart
        plt.figure(figsize=(10, 8))
        plt.barh(
            y=np.array(feature_names)[sorted_indices],
            width=np.array(importance_values)[sorted_indices],
            color='#4f46e5'
        )
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            plt.show()