"""
Contains all the available activation functions followed by their respective derivatives.
Derivatives share the same name as their parent function, plus an '_der'
"""

import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1 + np.exp(-x))

def sigmoid_der(x: np.ndarray) -> np.ndarray:
    return np.exp(-x) / ((np.exp(-x) + 1.) ** 2)

def softmax(x: np.ndarray) -> np.ndarray:
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)

def softmax_der(x: np.ndarray) -> np.ndarray:
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0) * (1. - exps / np.sum(exps, axis=0))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_der(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2