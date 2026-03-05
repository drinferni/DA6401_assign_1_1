"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from .activations import get_activation

class DenseLayer:
    def __init__(self, fan_in, fan_out, activation_name, init_type='xavier'):
        # Weight Initialization
        if init_type == 'xavier':
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))
        else: # Random
            self.W = np.random.randn(fan_in, fan_out) * 0.01
            
        self.b = np.zeros((1, fan_out))
        self.activation, self.activation_grad = get_activation(activation_name)
        
        # Cache for backprop
        self.input_cache = None
        self.z_cache = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        self.input_cache = X
        self.z_cache = np.dot(X, self.W) + self.b
        if self.activation:
            return self.activation(self.z_cache)
        return self.z_cache # Output layer (logits)

    def backward(self, delta):
        # delta is dL/dz
        batch_size = self.input_cache.shape[0]
        self.grad_W = np.dot(self.input_cache.T, delta) / batch_size
        self.grad_b = np.sum(delta, axis=0, keepdims=True) / batch_size
        
        # dL/dX for the previous layer
        return np.dot(delta, self.W.T)