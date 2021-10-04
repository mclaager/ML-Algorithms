# All supported weight initialization methods are stored here. The
# functions take the size of the current and previous layer, and return
# the initialized weights as numpy arrays.

import numpy as np

"""
All supported weight initialization methods are stored here. The
functions take the request shape (if required, the layer size as the first), and return
the initialized weights as numpy arrays.
"""
def zeros(shape: tuple) -> np.ndarray:
    return np.zeros(shape)

def ones(shape: tuple) -> np.ndarray:
    return np.zeros(shape)

def randu(shape: tuple) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.uniform(shape)

def randn(shape: tuple) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.standard_normal(shape)

def he(shape: tuple) -> np.ndarray:
    """Designed for ReLU() activaiton functions"""
    rng = np.random.default_rng()
    return rng.standard_normal(shape) * np.sqrt(2. / shape[0])

def xavier(shape: tuple) -> np.ndarray:
    """Designed for tanh() & sigmoid() activation functions (as these are approx linear around 0)"""
    rng = np.random.default_rng()
    return rng.standard_normal(shape) * np.sqrt(1. / shape[0])