import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import os

class DecisionBoundaryVisualizer:
    def __init__(self, model_debugger):
        self.model = model_debugger.model
        self.dataset = model_debugger.dataset
    
    def visualize_2d_decision_boundary(self, feature1_idx, feature2_idx, resolution=100, save_path=None):
        """Visualize the decision boundary for two selected features"""
        # Extract features
        X = np.array([x[0].numpy() for x, _ in self.dataset])
        y = np.array([y.numpy() for _, y in self.dataset])
        
        # Extract the two features we want to visualize
        X_2d = X[:, [feature1_idx, feature2_idx]]
        
        # Create a mesh grid
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # Create the mesh input (all combinations of the two features)
        mesh_input = np.c_[xx.ravel(), yy.ravel()]
        
        # For sklearn models
        if hasattr(self.model, 'predict'):
            Z = self.model.predict(mesh_input)
        # For PyTorch models
        elif isinstance(self.model, torch.nn.Module):
            self.model.eval()
            # Create tensor with same shape as original input but with only our two features
            mesh_tensor = torch.zeros((mesh_input.shape[0], X.shape[1]))
            # Set the two features we care about
            mesh_tensor[:, [feature1_idx, feature2_idx]] = torch.tensor(mesh_input, dtype=torch.float32)
            # Get predictions
            with torch.no_grad():
                Z = self.model(mesh_tensor).argmax(dim=1).numpy()
        
        # Reshape the predictions back to the mesh shape
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        
        plt.scatter(
            X_2d[:, 0], X_2d[:, 1], c=y, 
            cmap='coolwarm', edgecolors='k', s=20
        )
        
        plt.xlabel(f'Feature {feature1_idx}')
        plt.ylabel(f'Feature {feature2_idx}')
        plt.title('Decision Boundary')
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            plt.show()