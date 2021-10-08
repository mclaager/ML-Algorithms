class Layer():
    """
    A neural network can be divided into separate layers, where each layer
    must be able to forward and back propogate data. In each case,
    an input stream is required, back prop may require learning rate
    """
    def __init__(self) -> None:
        # The size of the input and output tensors
        self.input_size = None
        self.output_size = None

        # The tensors that are inputted and outputted by the layer
        self.input = None
        self.output = None

    def forward_prop(self, input):
        """Forward propagation through the layer"""
        raise NotImplementedError
    
    def backward_prop(self, delta, lr):
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
    
    def __repr__(self) -> str:
        """How the Layer is represented (all components to recreate the Layer)"""
        rtn_str = 'Layer input size: {}\n'.format(self.input_size)
        rtn_str += 'Layer output size : {}\n'.format(self.output_size)
        return rtn_str