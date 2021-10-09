# The optimizer structure is based off the following code: https://github.com/timvvvht/Neural-Networks-and-Optimizers-from-scratch

from neural_network.layers.layer import Layer
from utils.helper_functions import reraise

class Optimizer():
    """
    An optimizer takes trainable layers throughout the training process
    and performs computations to update the parameters in those layers
    """
    def __init__(self, lr: float = 0):
        self.lr = lr

    def set_lr(self, lr: float):
        self.lr = lr
    
    def optimize(self, layer: Layer, epoch: int = 0, time_step: int = 0):
        """Takes a layer and optimizes it according to """
        raise NotImplementedError

class SGD(Optimizer):
    """
    This is an unmodified form of stochastic gradient descent. The only necessary
    parameters here are the learning rate and the deltas that come within the layer.
    """
    def __init__(self, lr: float = 0.001):
        super().__init__(lr)
    
    def optimize(self, layer: Layer, epoch: int = 0, time_step: int = 0):
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
            param -= self.lr * value
