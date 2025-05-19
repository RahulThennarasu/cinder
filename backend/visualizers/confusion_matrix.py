import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

class ConfusionMatrixWithExamples:
    def __init__(self, model_debugger):
        self.model_debugger = model_debugger
        
    def generate_annotated_confusion_matrix(self, num_examples=3, save_path=None):
        """Generate a confusion matrix with example images for each cell"""
        # Get predictions and ground truth
        predictions = self.model_debugger.predictions
        ground_truth = self.model_debugger.ground_truth
        
        # Get raw data
        X = np.array([x[0].numpy() for x, _ in self.model_debugger.dataset])
        
        # Get confusion matrix
        matrix_data = self.model_debugger._calculate_confusion_matrix()
        cm = np.array(matrix_data["matrix"])
        labels = matrix_data["labels"]
        
        # Create a figure with a grid of subplots
        n_classes = len(labels)
        fig, axes = plt.subplots(
            n_classes, n_classes, 
            figsize=(3*n_classes, 3*n_classes),
            sharey=True, sharex=True
        )
        
        # For each cell in the confusion matrix
        for i in range(n_classes):
            for j in range(n_classes):
                # Find examples for this cell
                indices = np.where((ground_truth == i) & (predictions == j))[0]
                
                # If there are examples, display them
                if len(indices) > 0:
                    # Take up to num_examples
                    show_indices = indices[:min(num_examples, len(indices))]
                    
                    # For image data (assumes images)
                    if len(X.shape) > 2:  # If we have images (more than 2 dimensions)
                        # Create a montage of examples
                        examples = X[show_indices]
                        montage = self._create_montage(examples)
                        axes[i, j].imshow(montage, cmap='gray')
                        axes[i, j].set_title(f"{cm[i, j]}")
                    else:
                        # For non-image data, just show the count
                        axes[i, j].text(0.5, 0.5, f"{cm[i, j]}", 
                                      horizontalalignment='center',
                                      verticalalignment='center',
                                      fontsize=18)
                
                # Remove axis ticks
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                
                # Add grid
                axes[i, j].spines['top'].set_visible(True)
                axes[i, j].spines['right'].set_visible(True)
                axes[i, j].spines['bottom'].set_visible(True)
                axes[i, j].spines['left'].set_visible(True)
        
        # Add row and column labels
        for i, label in enumerate(labels):
            axes[i, 0].set_ylabel(f'True: {label}', fontsize=14)
            axes[0, i].set_xlabel(f'Pred: {label}', fontsize=14)
        
        # Add overall title
        plt.suptitle('Confusion Matrix with Examples', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            plt.show()
            
    def _create_montage(self, images, n_cols=None):
        """Create a montage from a list of images"""
        if n_cols is None:
            n_cols = min(len(images), 3)
            
        n_rows = (len(images) + n_cols - 1) // n_cols
        
        # Assume square images
        height, width = images[0].shape[:2]
        
        montage = np.zeros((height * n_rows, width * n_cols, *images[0].shape[2:]))
        
        for i, image in enumerate(images):
            row = i // n_cols
            col = i % n_cols
            
            montage[row*height:(row+1)*height, col*width:(col+1)*width] = image
            
        return montage