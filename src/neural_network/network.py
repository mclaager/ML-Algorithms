import numpy as np

from neural_network.layers.layer import Layer
from neural_network.layers.activation_layer import ActivationLayer

import ml_functions.cost_functions as cost_functions
from ml_functions.metrics import binary_accuracy


class NeuralNetwork():
    def __init__(self, *args: Layer) -> None:
        # Checks that there is at least 2 layers given as input
        assert len(args) >= 2, "Neural Network requires at least 2 layers as input."

        self.layers = list(args)

        self.cost_function = cost_functions.mse
        self.cost_function_der = cost_functions.mse_der

        self.metric = binary_accuracy
    
    def add(self, layer: Layer) -> None:
        self.layers.append(layer)
    
    def forward_prop(self, input: np.ndarray) -> np.ndarray:
        """Completes a forward pass of the network on the input 'inpt'"""
        output = input
        for layer in self.layers:
            output = layer.forward_prop(output)
        
        return output
    
    def backward_prop(self, expected: np.ndarray, output: np.ndarray, lr: float) -> list:
        """
        Completes a backward pass (backpropagation) using the true output 'expected'
        and previosuly attained network output 'output'. In most cases, to get the
        expected result, forward_prop() must be called directly before to properly set
        inputs for each layer.
        """
        error = self.cost_function_der(expected, output)
        for layer in reversed(self.layers):
            error = layer.backward_prop(error, lr)
        
        return error
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Uses the neural network to predict on an array of inputs.
        
        :param input_data: A numpy array where the first dimension holds
        each input, i.e. performing input_data.shape[0] should give the amount of inputs
        """
        # Amount of samples
        sample_count = input_data.shape[0]

        result = []

        for i in range(sample_count):
            # forward propogate 
            output = self.forward_prop(input_data[i])
            result.append(output)
        
        return np.array(result)

    def fit(self, X_train, y_train, epochs, lr, validation_data = None, verbose: bool = True) -> dict:
        """
        Trains the neural network using an array of inputs X_train and their labels y_train.

        :param X_train: Training data, a numpy array where the first dimension should be
        the total number of inputs
        :param y_train: Training labels, a numpy array where the first dimension should be
        the total number of inputs
        :param epochs: The number of epochs to train for
        :param lr: The learning rate of the model
        :param validation_data: Allows for inclusion of validation data that will be recorded,
        tuple of validation inputs and labels (optional)
        :param verbose: Whether to print out training data

        :returns: A dictionary of accuracies for training and validation, if applicable
        """

        sample_count = X_train.shape[0]

        # Checks that the training data is the same size as the labels
        if sample_count != y_train.shape[0]:
            raise IndexError('The training inputs are not the same size as the labels \
                ({} and {}, respectively)'.format(sample_count,y_train.shape[0]))
        
        history = {
            "accuracy" : []
        }
        
        # Checks sizes of validation data, if applicable
        if validation_data is not None:
            X_val = validation_data[0]
            y_val = validation_data[1]
            val_sample_count = X_val.shape[0]

            if val_sample_count != y_val.shape[0]:
                raise IndexError('The validation inputs are not the same size as the labels \
                ({} and {}, respectively)'.format(val_sample_count,y_val.shape[0]))
            
            history["val_accuracy"] = []


        if verbose:
            print('Training Network on {} sample{} for {} epoch{}...\
                '.format(sample_count, '' if sample_count == 1 else 's', epochs, '' if epochs == 1 else 's'))

        for i in range(epochs):
            loss = 0

            # trains the network on the training data
            for j in range(sample_count):
                # forward propogation, using jth input
                output = self.forward_prop(X_train[j])
                # Calculates loss
                loss += self.cost_function(y_train[j], output)
                # backward propogation
                self.backward_prop(y_train[j], output, lr)
            
            # Gets metrics for history
            metric_output = self.evaluate(X_train, y_train)
            history["accuracy"].append(metric_output)
            if validation_data is not None:
                metric_output = self.evaluate(X_val, y_val)
                history["val_accuracy"].append(metric_output)
            
            # Gets average loss
            loss /= sample_count
            if verbose:
                print('Epoch {}/{}:  loss = {}'.format(i+1,epochs,loss))
        
        return history

    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluates the model on testing data and labels
        
        :param X_test: The input for the testing data
        :param y_test: The labels for the testing data

        :returns: A value from the metric that was performed on the data
        """
        prediction = self.predict(X_test)
        return self.metric(prediction, y_test)

    def print_network(self) -> None:
        """Prints out a readble version of the network"""
        print('---------------------')
        print('NEURAL NETWORK PARAMS')
        print('---------------------\n')\
        
        if len(self.layers) == 0:
            print('Empty Network\n')
            return

        print('Layer 0 (Input Layer):')
        print('---------------------')
        print('  input size: {}\n'.format(self.layers[0].get_input_size()))
        
        l_count = 1
        for l in range(len(self.layers)):
            layer = self.layers[l]
            # Activation layers already taken care of in previous layer
            if isinstance(layer, ActivationLayer):
                continue

            print('Layer {}:'.format(l_count))
            print('---------')
            
            print('  layer size: {}'.format(layer.get_size()))
            # Prints the activation function of next layer for current layer (if applicable)
            if not l == len(self.layers)-1:
                next_layer = self.layers[l+1]
                if isinstance(next_layer, ActivationLayer):
                    print('  layer activation function: {}'.format(next_layer.activation_name))
            # Prints weights and biases of current layer
            f_weights = '    ' + str(layer.get_weights()).replace('\n', '\n    ')
            print('  layer weights:\n{}'.format(f_weights))
            f_biases = '    ' + str(layer.get_biases()).replace('\n', '\n    ')
            print('  layer biases:\n{}\n'.format(f_biases))
            
            l_count += 1