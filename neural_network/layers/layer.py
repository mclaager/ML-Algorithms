class Layer():
    """
    A neural network can be divided into separate layers, where each layer
    must be able to forward and back propogate data. Every layer has an
    input and output.
    """
    def __init__(self) -> None:
        # The size of the input and output tensors
        self.input_size = None
        self.output_size = None

        # The tensors that are inputted and outputted by the layer
        self.input = None
        self.output = None

        # Errors for any trainable parameters of the layer (if applicable)
        # NOTE: the names for the keys should match the name of the class variable to be updated
        self.deltas = {}

        # Whether or not the layer has trainable parameters
        self.trainable = False


    def forward_prop(self, input):
        """Forward propagation through the layer"""
        raise NotImplementedError
    
    def backward_prop(self, delta):
        """Back propagation through the layer"""
        raise NotImplementedError
    
    def get_input_size(self):
        """Returns the input size for the layer"""
        return self.input_size

    def get_output_size(self):
        """Returns the output size for the layer"""
        return self.output_size
    
    def get_input(self):
        """Returns the input for the layer"""
        return self.input

    def get_output(self):
        """Returns the output for the layer"""
        return self.output
    
    def get_deltas(self):
        """Returns the output for the layer"""
        return self.deltas
    
    def is_trainable(self):
        """Returns whether the layer has trainable parameters"""
        return self.trainable
    
    def __repr__(self) -> str:
        """How the Layer is represented (all components to recreate the Layer)"""
        rtn_str = 'Layer input size: {}\n'.format(self.input_size)
        rtn_str += 'Layer output size : {}\n'.format(self.output_size)
        return rtn_str