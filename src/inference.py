"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
import json
from utils.data_loader import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Assuming your NeuralNetwork class is in ann/neural_network.py
from ann.neural_network import NeuralNetwork

def parse_arguments():
    """
    Parse command-line arguments. 
    Matches train.py arguments as per instruction 1.2.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    # Dataset and Model Path
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-msp','--model_path', type=str, default='src/best_model.npy', help='Path to .npy weights')
    
    # Architecture (Must match the trained model)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-nhl', '--num_layers', type=int, default=1)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128])
    parser.add_argument('-a', '--activation', type=str, default='relu', choices=['sigmoid', 'tanh', 'relu'])
    
    # Arguments required to keep CLI consistent with train.py
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-o', '--optimizer', type=str, default='sgd')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-wi', '--weight_init', type=str, default='xavier')
    parser.add_argument('-wp', '--wandb_project', type=str, default='assignment_1')

    return parser.parse_args()


def load_model_weights(model_path):
    """
    Load trained model weights from disk as per PDF instructions.
    """
    try:
        data = np.load(model_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading model at {model_path}: {e}")
        return None


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data and compute required metrics.
    """
    # Get raw logits from forward pass
    logits = model.forward(X_test)
    
    # Convert logits to class predictions
    y_pred = np.argmax(logits, axis=1)
    y_test = np.argmax(y_test,axis=1)
    # print(y_pred,y_test)
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    
    # Note: Loss calculation would require the labels to be one-hot
    # but the primary output requirement is the 4 metrics below.
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
    
    # 1. Load Data

    _, _, X_test, y_test = load_data(args.dataset)


    # Preprocess: Flatten and normalize
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    
    # 2. Initialize Model Architecture
    model = NeuralNetwork(args)
    
    # 3. Load and set weights
    weights = load_model_weights(args.model_path)
    if weights is not None:
        model.set_weights(weights)
    else:
        print("Failed to initialize weights. Exiting.")
        return

    # 4. Run Evaluation
    results = evaluate_model(model, X_test, y_test)
    
    # Print Results as required by Section 1.1
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