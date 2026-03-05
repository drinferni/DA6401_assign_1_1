import numpy as np

class DenseLayer:
    def __init__(self, fan_in, fan_out):
        # Weight initialization is handled by the network class 
        # based on the weight_init argument
        self.W = None 
        self.b = None
        
        # Required by Section 1.1: Must expose these after backward()
        self.grad_W = None
        self.grad_b = None
        
        # Caches for backprop
        self.input_cache = None
        self.z_cache = None

    def forward(self, X):
        self.input_cache = X
        self.z_cache = np.dot(X, self.W) + self.b
        return self.z_cache

    def backward(self, delta):
        """
        delta: dL/dz for this layer
        Returns: dL/dX to be passed to the previous layer's activation
        """
        batch_size = delta.shape[0]
        
        # Compute gradients for this layer's parameters
        self.grad_W = np.dot(self.input_cache.T, delta) 
        self.grad_b = np.sum(delta, axis=0, keepdims=True)
        
        # Gradient to propagate back: dL/dX = delta * W^T
        return np.dot(delta, self.W.T)