"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
# import numpy as np
# import idx2numpy
# import numpy as np
# import os


# """
# Data Loading and Preprocessing
# Handles MNIST and Fashion-MNIST datasets
# """

def load_test_data():
    X_train = [[1.0, 2.0]]
    y_train = [0]

    X_test = [[5.0, 6.0]]
    y_test = [0]

    return X_train, y_train,X_test, y_test



# def load_data(dataset):
#     path = f"./data/{dataset}/raw"
    
#     x_train = idx2numpy.convert_from_file(os.path.join(path, 'train-images-idx3-ubyte'))
#     y_train_raw = idx2numpy.convert_from_file(os.path.join(path, 'train-labels-idx1-ubyte'))
    
#     x_test = idx2numpy.convert_from_file(os.path.join(path, 't10k-images-idx3-ubyte'))
#     y_test_raw = idx2numpy.convert_from_file(os.path.join(path, 't10k-labels-idx1-ubyte'))

#     x_train = x_train.reshape(-1, 784).astype('float32')/255
#     x_test = x_test.reshape(-1, 784).astype('float32') /255

#     y_train = np.eye(10)[y_train_raw.astype(int)]
#     y_test = np.eye(10)[y_test_raw.astype(int)]

#     return x_train, y_train, x_test, y_test


import numpy as np
from tensorflow import keras

def load_data(dataset):
    """
    Loads MNIST or Fashion-MNIST using Keras, processes them, 
    and splits the training data using Scikit-learn.
    """
    # 1. Load data from Keras
    if dataset.lower() == 'mnist':
        (x_train, y_train_raw), (x_test, y_test_raw) = keras.datasets.mnist.load_data()
    else:
        (x_train, y_train_raw), (x_test, y_test_raw) = keras.datasets.fashion_mnist.load_data()

    # 2. Reshape and Normalize
    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    x_test = x_test.reshape(-1, 784).astype('float32') / 255

    # 3. One-hot encode labels
    y_train = np.eye(10)[y_train_raw.astype(int)]
    y_test = np.eye(10)[y_test_raw.astype(int)]

    return x_train, y_train, x_test, y_test