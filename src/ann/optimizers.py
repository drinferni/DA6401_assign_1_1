"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.lr = learning_rate
        self.wd = weight_decay

    def update(self, layers, grad_Ws, grad_bs):
        raise NotImplementedError

class SGD(Optimizer):
    def update(self, layers, grad_Ws, grad_bs):
        # grad_Ws[0] is last layer, layers[-1] is last layer
        for i, layer in enumerate(reversed(layers)):
            dw = grad_Ws[i] + self.wd * layer.W
            db = grad_bs[i]
            layer.W -= self.lr * dw
            layer.b -= self.lr * db

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, weight_decay=0.0, momentum=0.9):
        super().__init__(learning_rate, weight_decay)
        self.m = momentum
        self.vW = None
        self.vb = None

    def update(self, layers, grad_Ws, grad_bs):
        if self.vW is None:
            self.vW = [np.zeros_like(g) for g in grad_Ws]
            self.vb = [np.zeros_like(g) for g in grad_bs]
        
        for i, layer in enumerate(reversed(layers)):
            dw = grad_Ws[i] + self.wd * layer.W
            self.vW[i] = self.m * self.vW[i] + self.lr * dw
            self.vb[i] = self.m * self.vb[i] + self.lr * grad_bs[i]
            layer.W -= self.vW[i]
            layer.b -= self.vb[i]

class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.001, weight_decay=0.0, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.eps = epsilon
        self.vW = None
        self.vb = None

    def update(self, layers, grad_Ws, grad_bs):
        if self.vW is None:
            self.vW = [np.zeros_like(g) for g in grad_Ws]
            self.vb = [np.zeros_like(g) for g in grad_bs]
        
        for i, layer in enumerate(reversed(layers)):
            dw = grad_Ws[i] + self.wd * layer.W
            self.vW[i] = self.beta * self.vW[i] + (1 - self.beta) * (dw**2)
            self.vb[i] = self.beta * self.vb[i] + (1 - self.beta) * (grad_bs[i]**2)
            layer.W -= (self.lr / (np.sqrt(self.vW[i]) + self.eps)) * dw
            layer.b -= (self.lr / (np.sqrt(self.vb[i]) + self.eps)) * grad_bs[i]