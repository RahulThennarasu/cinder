import numpy as np
import json
import os
import pickle
from sklearn.preprocessing import StandardScaler
import argparse

class LinUCBOptimizer:
    """Linear UCB implementation for compiler optimization selection"""
    
    def __init__(self, n_features, n_actions=4, alpha=1.0):
        """
        Initialize the LinUCB optimizer
        
        Args:
            n_features (int): Number of features in the context vector
            n_actions (int): Number of possible actions (optimization levels)
            alpha (float): Exploration parameter
        """
        self.n_features = n_features
        self.n_actions = n_actions  # Usually 4 for O0-O3
        self.alpha = alpha
        
        # Initialize model parameters for each action
        self.A = [np.identity(n_features) for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]
        self.theta = [np.zeros(n_features) for _ in range(n_actions)]
        
        # Track rewards for each action
        self.action_rewards = [[] for _ in range(n_actions)]
        self.action_counts = [0] * n_actions
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
    
    def select_action(self, features):
        """
        Select the best optimization level based on code features
        
        Args:
            features (numpy.ndarray): Feature vector of code
            
        Returns:
            int: The selected optimization level (0-3 for O0-O3)
        """
        # Scale features if scaler is fitted
        if self.is_scaler_fitted:
            features = self.scaler.transform(features.reshape(1, -1)).flatten()
        
        ucb_values = []
        
        for a in range(self.n_actions):
            # Compute the estimated reward
            self.theta[a] = np.linalg.solve(self.A[a], self.b[a])
            reward_estimate = np.dot(self.theta[a], features)
            
            # Compute the confidence bound
            cb = self.alpha * np.sqrt(
                np.dot(features.T, np.linalg.solve(self.A[a], features))
            )
            
            # Upper confidence bound
            ucb = reward_estimate + cb
            ucb_values.append(ucb)
            
        # Return the action with highest UCB value
        return np.argmax(ucb_values)
    
    def update(self, features, action, reward):
        """
        Update the model based on observed reward
        
        Args:
            features (numpy.ndarray): Feature vector of code
            action (int): The action taken (optimization level used)
            reward (float): The observed reward (performance improvement)
        """
        # Scale features if scaler is fitted
        if self.is_scaler_fitted:
            features = self.scaler.transform(features.reshape(1, -1)).flatten()
        
        # Update action statistics
        self.action_rewards[action].append(reward)
        self.action_counts[action] += 1
        
        # Update model parameters
        self.A[action] += np.outer(features, features)
        self.b[action] += reward * features
    
    def fit_scaler(self, features_list):
        """
        Fit the feature scaler on a set of feature vectors
        
        Args:
            features_list (list): List of feature vectors
        """
        self.scaler.fit(np.array(features_list))
        self.is_scaler_fitted = True
    
    def save(self, filepath):
        """
        Save the model to a file
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'A': self.A,
            'b': self.b,
            'theta': self.theta,
            'n_features': self.n_features,
            'n_actions': self.n_actions,
            'alpha': self.alpha,
            'action_rewards': self.action_rewards,
            'action_counts': self.action_counts,
            'scaler': self.scaler,
            'is_scaler_fitted': self.is_scaler_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath):
        """
        Load the model from a file
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            LinUCBOptimizer: The loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(model_data['n_features'], model_data['n_actions'], model_data['alpha'])
        model.A = model_data['A']
        model.b = model_data['b']
        model.theta = model_data['theta']
        model.action_rewards = model_data['action_rewards']
        model.action_counts = model_data['action_counts']
        model.scaler = model_data['scaler']
        model.is_scaler_fitted = model_data['is_scaler_fitted']
        
        return model


def extract_feature_vector(features_dict):
    """
    Convert a dictionary of features to a numpy array
    
    Args:
        features_dict (dict): Dictionary of features
        
    Returns:
        numpy.ndarray: Feature vector
    """
    # Define the order of features
    feature_names = [
        'basic_block_count',
        'instruction_count',
        'branch_count',
        'load_count',
        'store_count',
        'arithmetic_count',
        'call_count',
        'phi_count',
        'branch_density',
        'memory_density',
        'arithmetic_density',
        'avg_insts_per_bb'
    ]
    
    return np.array([features_dict.get(name, 0.0) for name in feature_names])


def main():
    """Main function for training and testing the model"""
    parser = argparse.ArgumentParser(description='Train the ML optimization model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--benchmark-dir', type=str, default='benchmarks', 
                        help='Directory containing benchmark programs')
    parser.add_argument('--model-path', type=str, default='data/model.pkl',
                        help='Path to save/load the model')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations')
    
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    if args.train:
        print("Training the ML optimization model...")
        
        # Initialize the model
        n_features = 12  # Number of features we extract
        model = LinUCBOptimizer(n_features)
        
        # Collect features from benchmark programs
        feature_vectors = []
        
        # TODO: In a real implementation, this would loop through benchmark programs,
        # extract features, and measure performance with different optimization levels.
        # For now, we'll simulate this process with random data.
        
        for _ in range(args.iterations):
            # Simulate feature extraction
            features = np.random.rand(n_features)
            feature_vectors.append(features)
            
            # Select action using the model
            action = model.select_action(features)
            
            # Simulate performance measurement
            # Higher optimization levels generally give better performance
            # but with some randomness to simulate real-world scenarios
            base_reward = [0.1, 0.5, 0.8, 1.0]  # Expected rewards for O0-O3
            reward = base_reward[action] + 0.2 * np.random.randn()
            
            # Update the model
            model.update(features, action, reward)
        
        # Fit the scaler on all feature vectors
        model.fit_scaler(feature_vectors)
        
        # Save the trained model
        model.save(args.model_path)
        print(f"Model saved to {args.model_path}")
        
        # Print statistics
        print("\nTraining Statistics:")
        for i in range(model.n_actions):
            count = model.action_counts[i]
            avg_reward = np.mean(model.action_rewards[i]) if count > 0 else 0
            print(f"O{i}: selected {count} times, avg reward: {avg_reward:.4f}")
    
    else:
        print("Testing mode (use --train to train the model)")
        
        # Check if model exists
        if not os.path.exists(args.model_path):
            print(f"Model not found at {args.model_path}. Run with --train first.")
            return
        
        # Load the model
        model = LinUCBOptimizer.load(args.model_path)
        print("Model loaded successfully")
        
        # Simple test with random features
        test_features = np.random.rand(model.n_features)
        action = model.select_action(test_features)
        print(f"For test features, recommended optimization level: O{action}")


if __name__ == "__main__":
    main()
