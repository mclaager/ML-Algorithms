# The optimizer structure is based off the following code: https://github.com/timvvvht/Neural-Networks-and-Optimizers-from-scratch

from neural_network.layers.layer import Layer
from tools.helper_functions import reraise

"""
An optimizer takes trainable layers throughout the training process
and performs computations to update the parameters in those layers
"""
    
def sgd(layer: Layer, lr: float = 0.01, epoch: int = 0, time_step: int = 0):
    """
    This is an unmodified form of stochastic gradient descent. The only necessary
    parameters here are the learning rate and the deltas that come within the layer.
    """
    deltas = layer.get_deltas()
    for key, value in deltas.items():
        # Gets the layer's parameter to be optimized
        try:
            param = getattr(layer, key)
        except AttributeError as err:
            reraise(err, "Layer malformed. Got error term for '{}', but it is not a parameter for the layer."\
                .format(key))
        # Optimizes that parameter
        param -= lr * value
