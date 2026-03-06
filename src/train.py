"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import wandb
import json
from utils.data_loader import *

from ann.neural_network import NeuralNetwork

# need to change the default values

def parse_arguments():

    parser = argparse.ArgumentParser(description='Train a modular MLP using NumPy')

    # Dataset and Logging
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-w_p', '--wandb_project', type=str, default='DA6401_Assignment1')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-o', '--optimizer', type=str, default='sgd',choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument("-nhl","--num_layers",dest="num_hidden_layers")
    parser.add_argument("-sz", "--hidden_size", dest="hidden_layer_sizes", nargs="+",  type=int, default=32)
    parser.add_argument('-a', '--activation', type=str, default='relu', choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-wi', '--weight_init', type=str, default='xavier', choices=['random', 'xavier'])
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'])
    parser.add_argument('-msp','--model_save_path', type=str, default='src/best_model.npy')

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

    config_dict = vars(args)
    with open("best_config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    
    print("training_complete")

if __name__ == '__main__':
    main()