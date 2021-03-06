import numpy as np

from neural_network.layers.layer import Layer

import utils.init_methods as init_methods
from utils.helper_functions import str_or_func

class DenseLayer(Layer):
    """
    A fully connected layer, where every neuron in the layer has a set of
    weights that connects it to other layers.
    """
    def __init__(self, input_size: int, output_size: int, weights_init_method: str or function = 'xavier',\
        biases_init_method: str or function = 'xavier') -> None:
        """
        Create a layer object with different inputs

        :param input_size: The amount of incomming connections
        :param output_size: The amount of pre-activated neuron outputs (true size of the layer)
        :param weights_init_method: The method used for weight initialization
        :param biases_init_method: The method used for bias initialization
        """
        super().__init__()
        self.trainable = True

        self.input_size = input_size
        self.output_size = output_size

        # Get the weight/bias initialization method
        self.weights_init_method = str_or_func(module=init_methods, identifier=weights_init_method,\
            err_msg="Invalid weight initialization method given.")
        self.biases_init_method = str_or_func(module=init_methods, identifier=biases_init_method,\
            err_msg="Invalid bias initialization method given.")
        
        # Initialize the weights and biases
        self.weights = self.weights_init_method((self.input_size, self.output_size))
        self.biases = self.biases_init_method((1, self.output_size))

        # Layer input and pre-activated neuron outputs
        self.input = init_methods.zeros((1, self.input_size))
        self.output = init_methods.zeros((1, self.output_size))

    def forward_prop(self, input: np.ndarray) -> np.ndarray:
        """Performs forward propagation through the layer"""
        # Saves input for back prop
        self.input = input
        # Updates the neurons for the layer
        self.output = input @ self.weights + self.biases
        # Returns pre-activated neurons
        return self.output
    
    def backward_prop(self, delta: np.ndarray) -> np.ndarray:
        """Performs a backward pass through the layer using aggregated errors from future layers 'delta'"""
        # Error for the weights and biases are the partial der. for each respecting term mul. by delta
        self.deltas['weights'] = self.input.T @ delta
        self.deltas['biases'] = delta

        # Return the change in error w.r.t. the input (delta for previous layer)
        return delta @ self.weights.T
    
    def get_weights(self) -> np.ndarray:
        """Returns the weights of the layer"""
        return self.weights

    def get_biases(self) -> np.ndarray:
        """Returns the biases of the layer"""
        return self.biases
    
    def __repr__(self) -> str:
        rtn_str = super().__repr__()
        f_weights = '  ' + str(self.get_weights()).replace('\n', '\n  ')
        rtn_str += 'Layer weights:\n{}\n'.format(f_weights)
        f_biases = '  ' + str(self.get_biases()).replace('\n', '\n  ')
        rtn_str += 'Layer biases:\n{}'.format(f_biases)
        return rtn_str