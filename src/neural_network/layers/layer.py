class Layer():
    """
    A neural network can be divided into separate layers, where each layer
    must be able to forward and back propogate data. In each case,
    an input stream is required, back prop may require learning rate
    """
    def forward_prop(self, input):
        """Forward propagation through the layer"""
        raise NotImplementedError
    
    def backward_prop(self, delta, lr):
        """Back propagation through the layer"""
        raise NotImplementedError