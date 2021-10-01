"""
Contains different cost functions & their respective derivatives used for error calculation.
Derivatives share the same name as their parent function, plus an '_der'
"""

import numpy as np

def mse(expected: np.ndarray, output: np.ndarray):
    """Mean-Squared Error"""
    return np.mean((expected - output) ** 2)

def mse_der(expected: np.ndarray, output: np.ndarray):
    return 2 * (output - expected) / expected.size