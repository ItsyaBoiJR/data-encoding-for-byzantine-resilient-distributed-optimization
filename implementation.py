import numpy as np
import torch
from torch import nn, optim

class ByzantineResilientPGD:
    def __init__(self, m, t, learning_rate=0.01):
        self.m = m  # Total number of workers
        self.t = t  # Maximum number of Byzantine workers
        self.learning_rate = learning_rate

    def encode_data(self, data):
        """
        Encodes the data using a sparse encoding scheme.
        This is a simple example of encoding by adding redundancy.
        """
        n, d = data.shape
        redundancy = 2 * self.t
        encoded_data = np.zeros((self.m, d))
        
        for i in range(self.m):
            if i < n:
                encoded_data[i] = data[i]
            else:
                # Add redundancy by summing rows
                encoded_data[i] = np.sum(data[:redundancy], axis=0)
        
        return encoded_data

    def decode_gradients(self, gradients):
        """
        Decodes gradients using majority voting to mitigate Byzantine workers.
        """
        gradients = np.array(gradients)
        decoded_gradient = np.median(gradients, axis=0)  # Median is robust to outliers
        return decoded_gradient

    def proximal_gradient_descent(self, X, y, initial_params, epochs=100):
        """
        Perform Byzantine-resilient Proximal Gradient Descent.
        """
        params = initial_params
        n, d = X.shape
        
        # Split data among workers
        data_splits = np.array_split(X, self.m, axis=0)
        label_splits = np.array_split(y, self.m, axis=0)
        
        for epoch in range(epochs):
            gradients = []
            
            for i in range(self.m):
                # Simulate Byzantine workers
                if i < self.t:
                    # Byzantine worker sends arbitrary gradients
                    gradient = np.random.randn(d)
                else:
                    # Honest worker computes the gradient
                    X_i = data_splits[i]
                    y_i = label_splits[i]
                    predictions = X_i @ params
                    gradient = -2 * X_i.T @ (y_i - predictions) / len(y_i)
                
                gradients.append(gradient)
            
            # Decode gradients to mitigate Byzantine workers
            robust_gradient = self.decode_gradients(gradients)
            
            # Update parameters using the robust gradient
            params = params - self.learning_rate * robust_gradient
        
        return params

if __name__ == '__main__':
    # Dummy data for testing
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic data
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    true_params = np.random.randn(n_features)
    y = X @ true_params + 0.1 * np.random.randn(n_samples)
    
    # Initialize ByzantineResilientPGD
    m = 10  # Total workers
    t = 3   # Byzantine workers
    learning_rate = 0.01
    initial_params = np.zeros(n_features)
    
    br_pgd = ByzantineResilientPGD(m, t, learning_rate)
    
    # Run Byzantine-resilient Proximal Gradient Descent
    final_params = br_pgd.proximal_gradient_descent(X, y, initial_params, epochs=100)
    print("True Parameters:", true_params)
    print("Estimated Parameters:", final_params)