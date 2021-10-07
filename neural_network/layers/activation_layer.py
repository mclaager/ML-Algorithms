from neural_network.layers.layer import Layer
import tools.activation_functions as activation_functions
from tools.helper_functions import str_or_func

class ActivationLayer(Layer):
    """
    Though in literature, where activations tend to be seen as part
    of a layer, it is useful to construct them as the non-linear aspects
    of a neural network. They are non-trainable, but can forward and
    back propagate through chosen function.
    """

    def __init__(self, activation_func: str or function,\
        activation_func_der: str or function = None) -> None:
        """
        Initailize the activation layer with either custom functions, string
        names of the function and its derivative, or just the string name of
        the function. Will also set the name for printing usage.

        :param activation_func: Defines what the activation function is. Either a string
        or function. If a string is given, then giving the derivative is most likely unnecessary
        :param activation_func_der: Defines the derivative of the activation function. Either a
        string or function. Required if first argument was not a string
        """

        af_der = activation_func_der

        # Test if only one input was given
        if af_der is None:
            if isinstance(activation_func, str):
                af_der = activation_func + '_der'
            else:
                raise TypeError('Activation function must be string type \
                    if derivative is not given explicitly.')
        
        
        # Select existing library activation function or use custom
        self.activation_func = str_or_func(module=activation_functions, identifier=activation_func,\
            err_msg="Invalid activation function given.")
        # Select existing library derivative or use custom
        self.activation_func_der = str_or_func(module=activation_functions, identifier=af_der,\
            err_msg="Invalid activation function derivative given.")
        
        # Set the name of the activation function
        if isinstance(activation_func, str):
            self.activation_name = activation_func
        else:
            self.activation_name = activation_func.__name__
    
    def forward_prop(self, input):
        """Forward propagation using the activation function"""
        # Saves the input array for use in back propagation
        self.input = input
        return self.activation_func(self.input)
    
    def backward_prop(self, delta, lr):
        """
        Backwards propagation using derivative of the activation function
        
        :param delta: The output error
        :param lr: Learning rate, unused as this is a non-learnable layer
        """
        return delta * self.activation_func_der(self.input)