"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np

#optimizer are implemented as classes so that they are hold values of previous iterations
# we will not have to manually do that in the neural network class

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

class NAG:
    def __init__(self, learning_rate=0.01, weight_decay=0, nag=0.9):
        self.lr = learning_rate
        self.wd = weight_decay
        self.m = nag  
        self.vW = None
        self.vb = None
        self.ori_W = None
        self.ori_b = None

    def exchange_wgt(self, layers):
        if self.vW is None:
            return
        
        self.ori_W = [np.zeros_like(g) for g in self.vW]
        self.ori_b = [np.zeros_like(g) for g in self.vb]

        for i, layer in enumerate(reversed(layers)):
            self.ori_W[i] = layer.W.copy()
            self.ori_b[i] = layer.b.copy()

            layer.W -= self.m * self.vW[i]
            layer.b -= self.m * self.vb[i]

    def update(self, layers, grad_Ws, grad_bs):
        if self.vW is None:
            self.vW = [np.zeros_like(g) for g in grad_Ws]
            self.vb = [np.zeros_like(g) for g in grad_bs]
    
        for i, layer in enumerate(reversed(layers)):
            dw = grad_Ws[i] + self.wd * layer.W
            db = grad_bs[i]
            
            self.vW[i] = self.m * self.vW[i] + self.lr * dw
            self.vb[i] = self.m * self.vb[i] + self.lr * db

            if self.ori_W is None:
                layer.W -= self.vW[i]
                layer.b -= self.vb[i]
            else :
                layer.W = self.ori_W[i] - self.vW[i]
                layer.b = self.ori_b[i] - self.vb[i]

        