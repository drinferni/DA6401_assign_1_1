import numpy as np
from .neural_layer import DenseLayer
from .activations import softmax,get_activation
from .optimizers import SGD, Momentum, RMSProp

class NeuralNetwork:
    def __init__(self, cli_args):
        self.args = cli_args
        self.layers = []
        
        # Architecture Setup
        dims = [784] + cli_args.hidden_layer_sizes + [10]
        self.act_fn, self.act_grad = get_activation(cli_args.activation)

        for i in range(len(dims) - 1):
            layer = DenseLayer(dims[i], dims[i+1])
            # Weight Initialization
            if cli_args.weight_init == 'xavier':
                limit = np.sqrt(6 / (dims[i] + dims[i+1]))
                layer.W = np.random.uniform(-limit, limit, (dims[i], dims[i+1]))
            else:
                layer.W = np.random.randn(dims[i], dims[i+1]) * 0.01
            layer.b = np.zeros((1, dims[i+1]))
            self.layers.append(layer)
        
        if cli_args.optimizer == 'sgd':
            self.optimizer = SGD(cli_args.learning_rate, cli_args.weight_decay)
        elif cli_args.optimizer == 'momentum':
            self.optimizer = Momentum(cli_args.learning_rate, cli_args.weight_decay)
        elif cli_args.optimizer == 'rmsprop':
            self.optimizer = RMSProp(cli_args.learning_rate, cli_args.weight_decay)
        # else:
        #     self.optimizer = NAG(cli_args.learning_rate, cli_args.weight_decay)

    def forward(self, X):
        out = X
        for i, layer in enumerate(self.layers):
            out = layer.forward(out)
            if i < len(self.layers) - 1: # No activation on output layer (logits)
                out = self.act_fn(out)
        return out

    def backward(self, y_true, y_pred_logits):
        n = y_true.shape[0]
        
        # CRITICAL FIX: Ensure y_true is one-hot
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            y_one_hot = np.zeros_like(y_pred_logits)
            y_one_hot[np.arange(n), y_true.ravel().astype(int)] = 1
        else:
            y_one_hot = y_true

        # Initial Delta (dL/dz) at the output layer
        if self.args.loss == 'cross_entropy':
            probs = softmax(y_pred_logits)
            delta = (probs - y_one_hot) / n # Mean Gradient
        else: # Mean Squared Error
            # MSE grad w.r.t logits: 2/n * (y_pred - y_true)
            delta = 2 * (y_pred_logits - y_one_hot) / n

        grad_W_list = []
        grad_b_list = []

        # Backpropagate through layers (Last to First)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            # 1. Update layer.grad_W and layer.grad_b via layer's internal backward
            # This returns dL/dX (gradient flowing into this layer's input)
            delta = layer.backward(delta)
            
            # 2. Collect gradients: index 0 must be the last layer
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

            # 3. Apply activation gradient of the PREVIOUS layer 
            # (unless we just processed the first layer)
            if i > 0:
                prev_z = self.layers[i-1].z_cache
                delta = delta * self.act_grad(prev_z)

        # Requirement: Store as object arrays
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i in range(len(grad_W_list)):
            self.grad_W[i] = grad_W_list[i]
            self.grad_b[i] = grad_b_list[i]

        return self.grad_W, self.grad_b
    
    def update_weights(self):
        # We pass self.grad_W/b (which are in last-to-first order) to optimizer
        self.optimizer.update(self.layers, self.grad_W, self.grad_b)

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        # Implementation of mini-batch loop
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for i in range(0, num_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                logits = self.forward(X_batch)
                self.backward(y_batch, logits)
                self.update_weights()

    def get_weights(self):
        # Ensure we return weights in the order they were created
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W
            d[f"b{i}"] = layer.b
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            layer.W = weight_dict[f"W{i}"]
            layer.b = weight_dict[f"b{i}"]