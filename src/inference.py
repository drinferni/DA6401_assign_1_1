"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
import wandb
from utils.data_loader import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ann.neural_network import NeuralNetwork

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-msp','--model_path', type=str, default='src/best_model.npy', help='Path to .npy weights')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument("-nhl","--num_layers")
    parser.add_argument("-sz", "--hidden_size", nargs="+",  type=int, default=32)
    parser.add_argument('-a', '--activation', type=str, default='relu', choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-o', '--optimizer', type=str, default='sgd')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-wi', '--weight_init', type=str, default='xavier')
    parser.add_argument('-w_p', '--wandb_project', type=str, default='assignment_1')

    return parser.parse_args()


def load_model_weights(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test): 

    logits = model.forward(X_test)
    

    y_pred = np.argmax(logits, axis=1)
    y_test = np.argmax(y_test,axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    
    metrics = {
        "logits": logits,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    return metrics


def main():
    args = parse_arguments()
    
    _, _, X_test, y_test = load_data(args.dataset)

    wandb.init(project=args.wandb_project, config=vars(args))

    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    
    model = NeuralNetwork(args)
    
    weights = load_model_weights(args.model_path)
    if weights is not None:
        model.set_weights(weights)
    else:
        print("Failed to initialize weights")
        return

    results = evaluate_model(model, X_test, y_test)

    wandb.log(results)
    print(results)

    wandb.finish()
    
    print("\n--- Evaluation Results ---")
    print(f"Dataset:   {args.dataset}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-score:  {results['f1_score']:.4f}")
    print("---------------------------\n")

    return results


if __name__ == '__main__':
    main()