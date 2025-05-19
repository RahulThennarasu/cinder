import numpy as np
import matplotlib.pyplot as plt
import os

class CalibrationVisualizer:
    def __init__(self, model_debugger):
        self.model_debugger = model_debugger
        
    def generate_calibration_curve(self, n_bins=10, save_path=None):
        """Generate a reliability diagram showing predicted vs actual probabilities"""
        if not hasattr(self.model_debugger, 'prediction_probas'):
            raise ValueError("Model debugger doesn't have probability predictions")
            
        # Get prediction probabilities and ground truth
        y_prob = self.model_debugger.prediction_probas
        y_true = self.model_debugger.ground_truth
        
        # For binary classification
        if y_prob.shape[1] == 2:
            # Use probability of class 1
            prob_pos = y_prob[:, 1]
            
            # Create bins and find bin edges
            bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
            binids = np.digitize(prob_pos, bins) - 1
            
            # Calculate actual probability in each bin
            bin_sums = np.bincount(binids, weights=prob_pos, minlength=len(bins))
            bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
            bin_total = np.bincount(binids, minlength=len(bins))
            
            # Avoid division by zero
            nonzero = bin_total != 0
            prob_true = bin_true[nonzero] / bin_total[nonzero]
            prob_pred = bin_sums[nonzero] / bin_total[nonzero]
            
            # Plot calibration curve
            plt.figure(figsize=(10, 8))
            plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
            plt.plot(prob_pred, prob_true, 's-', label='Model')
            
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Curve (Reliability Diagram)')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                return save_path
            else:
                plt.show()