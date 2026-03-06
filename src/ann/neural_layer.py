import numpy as np

class DenseLayer:
    def __init__(self, fan_in, fan_out):

        self.W = None 
        self.b = None
        
        self.grad_W = None
        self.grad_b = None
        
        self.input_cache = None
        self.z_cache = None

    def forward(self, X):
        self.input_cache = X
        self.z_cache = np.dot(X, self.W) + self.b
        return self.z_cache

    def backward(self, delta):

        self.grad_W = np.dot(self.input_cache.T, delta) 
        self.grad_b = np.sum(delta, axis=0, keepdims=True)
        
        # Gradient to propagate back: dL/dX = delta * W^T
        return np.dot(delta, self.W.T)