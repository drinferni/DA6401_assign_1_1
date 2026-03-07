import numpy as np
from .neural_layer import *
from .activations import *
from .optimizers import *

class NeuralNetwork:
    def __init__(self, cli_args):
        self.args = cli_args
        self.layers = []
        
        # Architecture Setup
        dims = [784] + cli_args.hidden_size + [10]
        self.act_fn, self.act_grad = get_activation(cli_args.activation)

        for i in range(len(dims) - 1):
            layer = DenseLayer(dims[i], dims[i+1])
            # Weight Initialization
            if cli_args.weight_init == 'xavier':
                limit = np.sqrt(6 / (dims[i] + dims[i+1]))
                layer.W = np.random.uniform(-limit, limit, (dims[i], dims[i+1]))
            elif cli_args.weight_init == 'zero':
                layer.W = np.zeros((dims[i],dims[i+1]))
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
        else:
            self.optimizer = NAG(cli_args.learning_rate, cli_args.weight_decay)

    def forward(self, X):
        out = X
        for i, layer in enumerate(self.layers):
            out = layer.forward(out)
            if i < len(self.layers) - 1: # No activation on output layer (logits)
                out = self.act_fn(out)
            # result = ",".join(map(str, out))
            # print(f"{self.args.activation},{e},{i},",result)
        return out
    
    def get_loss(self,y_true,y_pred_logits):
        n = y_true.shape[0]
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            y_one_hot = np.zeros_like(y_pred_logits)
            y_one_hot[np.arange(n), y_true.ravel().astype(int)] = 1
        else:
            y_one_hot = y_true

        if self.args.loss == 'cross_entropy':
            probs = softmax(y_pred_logits)
            delta = - (y_one_hot * np.log(probs + 1e-15)) / n
        else: # mse 
            delta = 0.5 * np.square(y_pred_logits - y_one_hot) / n
        
        return np.sum(delta)

        

    def backward(self, y_true, y_pred_logits,e):

        n = y_true.shape[0]
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            y_one_hot = np.zeros_like(y_pred_logits)
            y_one_hot[np.arange(n), y_true.ravel().astype(int)] = 1
        else:
            y_one_hot = y_true


        if self.args.loss == 'cross_entropy':
            probs = softmax(y_pred_logits)
            delta = (probs - y_one_hot) / n 
        else: 
            delta = 2 * (y_pred_logits - y_one_hot) / n

        grad_W_list = []
        grad_b_list = []

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            delta = layer.backward(delta)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

            # if i == 1:
            #     for x in range(0,5):
            #         print(f"{self.args.weight_init},{e},{x},{delta[0][x].mean()}")

            if i > 0:
                prev_z = self.layers[i-1].z_cache
                delta = delta * self.act_grad(prev_z)


        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i in range(len(grad_W_list)):
            self.grad_W[i] = grad_W_list[i]
            self.grad_b[i] = grad_b_list[i]

        return self.grad_W, self.grad_b
    
    def update_weights(self):
        self.optimizer.update(self.layers, self.grad_W, self.grad_b)

    def train(self, X_train, y_train, epochs, batch_size):
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            final_loss = 0
            for i in range(0, num_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                if self.args.optimizer == "nag":
                    self.optimizer.exchange_wgt(self.layers)
                logits = self.forward(X_batch)
                final_loss += self.get_loss(y_batch,logits)
                self.backward(y_batch, logits,epoch)
                # grad_norm_L1 = np.linalg.norm(self.layers[0].grad_W)
                # print(f"{self.args.hidden_size},{self.args.activation},{grad_norm_L1}")
                self.update_weights()
            # print(f"{self.args.loss},{epoch},{final_loss}")
        
    def get_weights(self):
        d = dict()
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W
            d[f"b{i}"] = layer.b
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            layer.W = weight_dict[f"W{i}"].copy()
            layer.b = weight_dict[f"b{i}"].copy()