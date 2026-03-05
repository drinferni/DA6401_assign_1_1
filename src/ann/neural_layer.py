"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from .activations import get_activation

class DenseLayer:
    def __init__(self, fan_in, fan_out, activation_name, init_type='xavier'):
        if init_type == 'xavier':
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))
        else:
            self.W = np.random.randn(fan_in, fan_out) * 0.01
            
        self.b = np.zeros((1, fan_out))
        
        # Placeholders for gradients required by Section 1.1
        self.grad_W = None
        self.grad_b = None
        
        # Caches for backprop
        self.x_cache = None
        self.z_cache = None

    def forward(self, X):
        self.x_cache = X
        self.z_cache = np.dot(X, self.W) + self.b
        return self.z_cache

    def backward_step(self, delta):
        """
        delta: dL/dz (gradient with respect to the linear output of THIS layer)
        Returns: dL/dx (gradient with respect to the input of THIS layer)
        """
        # 1. Compute gradients for this layer's parameters
        # Note: We do NOT divide by batch_size here if delta is already averaged
        self.grad_W = np.dot(self.x_cache.T, delta)
        self.grad_b = np.sum(delta, axis=0, keepdims=True)
        
        # 2. Compute gradient for the previous layer (dL/dx)
        return np.dot(delta, self.W.T)