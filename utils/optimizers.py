# The optimizer structure is based off the following code: https://github.com/timvvvht/Neural-Networks-and-Optimizers-from-scratch

from neural_network.layers.layer import Layer
from utils.helper_functions import reraise

class Optimizer():
    """
    An optimizer takes a trainable model (or model component) throughout the training process
    and performs computations to update the parameters in those components
    """
    def __init__(self, lr: float = 0):
        self.lr = lr

    def set_lr(self, lr: float):
        self.lr = lr
    
    def optimize(self, trainable, batch_size: float = None, epoch: int = 0, time_step: int = 0):
        """
        Takes a trainable model/component and optimizes the trainable parameters. Batch size, the current epoch count,
        and time step are parameters commonly used in optimizers.
        """
        raise NotImplementedError

class SGD(Optimizer):
    """
    This is an unmodified form of stochastic gradient descent. The only necessary
    parameters here are the learning rate and the deltas that come within the component.
    """
    def __init__(self, lr: float = 0.001):
        super().__init__(lr)
    
    def optimize(self, trainable, batch_size: float = 1., epoch: int = 0, time_step: int = 0):
        """Performs SGD"""
        deltas = trainable.get_deltas()
        for key, value in deltas.items():
            # Gets the layer's parameter to be optimized
            try:
                param = getattr(trainable, key)
            except AttributeError as err:
                reraise(err, "Trainable component malformed. Got error term for '{}', but it is not a \
                    parameter for the trainable.".format(key))
            # Optimizes that parameter
            param -= self.lr * (1./batch_size) * value
