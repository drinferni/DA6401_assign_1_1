import numpy as np
from .neural_layer import DenseLayer
from .activations import softmax
from .optimizers import SGD, Momentum, RMSProp

class NeuralNetwork:
    def __init__(self, cli_args):
        self.args = cli_args
        self.layers = []
        
        # Architecture Setup
        # Input layer (MNIST/Fashion-MNIST are 28x28 = 784)
        input_dim = 784 
        hidden_sizes = cli_args.hidden_size # Expects a list from argparse
        output_dim = 10 # 10 classes
        
        prev_dim = input_dim
        # Add Hidden Layers
        for size in hidden_sizes:
            self.layers.append(DenseLayer(prev_dim, size, cli_args.activation, cli_args.weight_init))
            prev_dim = size
            
        # Add Output Layer (no activation, returns logits)
        self.layers.append(DenseLayer(prev_dim, output_dim, None, cli_args.weight_init))

        # Optimizer Setup
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
        for layer in self.layers:
            out = layer.forward(out)
        return out # These are the logits

    def backward(self, y_true, y_pred_logits):
        """
        Calculates gradients for Cross-Entropy loss.
        y_true: one-hot encoded or integer labels (assumed one-hot for derivation)
        """
        batch_size = y_true.shape[0]
        probs = softmax(y_pred_logits)
        
        # Gradient of Cross-Entropy w.r.t Logits: (P - Y)
        delta = (probs - y_true) 
        
        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            delta = layer.backward(delta)
            
            # If not the first layer (going backwards), multiply by activation grad
            if i > 0:
                prev_layer_z = self.layers[i-1].z_cache
                delta = delta * self.layers[i-1].activation_grad(prev_layer_z)
            
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # Create explicit object arrays as per instructions
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

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
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()