from neural_network.layers.layer import Layer
import ml_functions.activation_functions as activation_functions
from ml_functions.helper_functions import reraise, str_or_func

class ActivationLayer(Layer):
    """
    Though in literature, where activations tend to be seen as part
    of a layer, it is useful to construct them as the non-linear aspects
    of a neural network. They are non-trainable, but can forward and
    back propagate through chosen function.
    """

    def __init__(self, activation_func: str or function,
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
                raise TypeError('Defining activation layer using a single argument \
                    requires a string name to be used.')
        
        
        # Select existing library activation function or use custom
        try:
            self.activation_func = str_or_func(activation_functions, activation_func)
        except AttributeError as err:
            reraise(err, '"{}" is not a valid activation function'.format(activation_func))
        # Select existing library derivative or use custom
        try:
            self.activation_func_der = str_or_func(activation_functions, af_der)
        except AttributeError as err:
            reraise(err, '"{}" is not a valid derivative to an activation function'.format(af_der))
        
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