"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import wandb
import json
from utils.data_loader import *
from sklearn.model_selection import train_test_split

# Assuming your NeuralNetwork class is in neural_network.py
from ann.neural_network import NeuralNetwork

def parse_arguments():
    """
    Parse command-line arguments as per Assignment Specifications.
    """
    parser = argparse.ArgumentParser(description='Train a modular MLP using NumPy')

    # Dataset and Logging
    parser.add_argument('-d', '--dataset', type=str, default='mnist', 
                        choices=['mnist', 'fashion_mnist'], help='Dataset to use')
    parser.add_argument('-wp', '--wandb_project', type=str, default='DA6401_Assignment1', 
                        help='Weights & Biases project name')
    
    # Hyperparameters
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('-o', '--optimizer', type=str, default='sgd', 
                        choices=['sgd', 'momentum', 'nag', 'rmsprop'], help='Optimizer choice')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='L2 regularization weight')
    
    # Architecture
    parser.add_argument("-nhl","--num_layers",dest="num_hidden_layers")
    parser.add_argument("-sz", "--hidden_size", dest="hidden_layer_sizes", nargs="+",  type=int)
    parser.add_argument('-a', '--activation', type=str, default='relu', 
                        choices=['sigmoid', 'tanh', 'relu'], help='Activation function')
    parser.add_argument('-wi', '--weight_init', type=str, default='xavier', 
                        choices=['random', 'xavier'], help='Weight initialization method')
    
    # Loss and Output
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', 
                        choices=['mean_squared_error', 'cross_entropy'], help='Loss function')
    
    # File Paths
    parser.add_argument('-msp','--model_save_path', type=str, default='src/best_model.npy', 
                        help='Relative path to save weights')

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Initialize W&B
    # wandb.init(project=args.wandb_project, config=vars(args))
    

    # Load and Preprocess Data
    x_train, y_train, x_test, y_test = load_data(args.dataset)

    # Initialize Model
    model = NeuralNetwork(args)

    print(f"Starting training on {args.dataset} using {args.optimizer}...")

    model.train(x_train, y_train, epochs=1, batch_size=args.batch_size)
        
    # wandb.finish()

    best_weights = model.get_weights()
    np.save("best_model.npy", best_weights)
    print("training_complete")

if __name__ == '__main__':
    main()