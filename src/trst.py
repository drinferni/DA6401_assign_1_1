import numpy as np
import argparse
from sklearn.metrics import f1_score
from ann.neural_network import NeuralNetwork

best_config= argparse.Namespace(
            dataset="mnist",
            epochs=25,
            batch_size=128,
            loss="cross_entropy",
            optimizer="rmsprop",
            weight_decay=0.001,
            learning_rate=0.001,
            num_layers=4,
            hidden_size=[128, 64, 32, 16],
            activation="relu",
            weight_init="xavier"
        )

model = NeuralNetwork(best_config)

weights = np.load("best_model.npy", allow_pickle=True).item()

model.set_weights(weights)

X_test = np.random.rand(100, 784)  # 100 samples, 784 features

y_true = np.random.randint(0, 10, size=(100,))  # 100 samples, 10 classes (0-9)

y_pred = model.forward(X_test)

y_pred_labels = np.argmax(y_pred, axis=1)

print("F1 Score:", f1_score(y_true, y_pred_labels, average='macro'))
