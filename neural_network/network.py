import numpy as np

from neural_network.layers.layer import Layer

import utils.loss_functions as loss_functions
import utils.optimizers as optimizers
import utils.metrics as metrics

from utils.helper_functions import coerce_1d_array, str_or_class, str_or_func


class NeuralNetwork():
    def __init__(self, layers: list) -> None:
        self.layers = layers
        self.optimizer = None
        self.loss_function = None
        self.loss_function_der = None
        self.metric = None
        self.metric_name = None
    
    def add(self, layer: Layer) -> None:
        """Adds a layer to the network"""
        self.layers.append(layer)
    
    def compile(self, optimizer: str or function = 'SGD', loss: str or function = 'mse',\
        loss_der: str or function = 'mse_der', metric: str or function = 'binary_accuracy'):
        """
        Compiles a model with a given optimizer, loss function (+ derivative), and
        metric for model evaluation. This must be called before training takes place.
        """
        # Gets an instance of the optimizer
        self.optimizer = str_or_class(module=optimizers, identifier=optimizer,\
            err_msg="Invalid optimizer given.")

        # Test if only one input was given for cost function
        lf_der = loss_der
        if lf_der is None:
            if isinstance(loss, str):
                lf_der = loss + '_der'
            else:
                raise TypeError('Loss function must be string type \
                    if derivative is not given explicitly.')

        # Set the cost function and its derivative
        self.loss_function = str_or_func(module=loss_functions, identifier=loss,\
            err_msg="Invalid loss function given.")
        self.loss_function_der = str_or_func(module=loss_functions, identifier=lf_der,\
            err_msg="Invalid loss function derivative given.")

        # Set the metric
        self.metric = str_or_func(module=metrics, identifier=metric,\
            err_msg="Invalid metric given.")
        # Sets the name of the metric
        if isinstance(metric, str):
            self.metric_name = metric
        else:
            self.metric_name = metric.__name__

    
    def forward_prop(self, input: np.ndarray) -> np.ndarray:
        """Completes a forward pass of the network on the input 'inpt'"""
        output = input
        for layer in self.layers:
            output = layer.forward_prop(output)
        
        # Flattens the output into a vector
        return output.flatten()
    
    def backward_prop(self, expected: np.ndarray, output: np.ndarray) -> list:
        """
        Completes a backward pass (backpropagation) using the true output 'expected'
        and previosuly attained network output 'output'. In most cases, to get the
        expected result, forward_prop() must be called directly before to properly set
        inputs for each layer.
        """
        error = self.loss_function_der(expected, output)
        for layer in reversed(self.layers):
            error = layer.backward_prop(error)
    
    def update_params(self, epoch: int, time_step: int):
        """
        Will update all the trainable layers in the network according to the optimizer.
        """
        for layer in self.layers:
            if layer.is_trainable():
                self.optimizer.optimize(layer, epoch, time_step)

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

        assert len(self.layers) > 0, "Network is empty."

        # Sets the learning rate for the optimizer
        self.optimizer.set_lr(lr)

        # Gets amount of training samples
        sample_count = X_train.shape[0]

        # Checks that the training data is the same size as the labels
        if sample_count != y_train.shape[0]:
            raise IndexError('The training inputs are not the same size as the labels \
                ({} and {}, respectively)'.format(sample_count,y_train.shape[0]))
        
        history = {
            self.metric_name : []
        }
        
        # Checks sizes of validation data, if applicable
        if validation_data is not None:
            X_val = validation_data[0]
            y_val = validation_data[1]
            val_sample_count = X_val.shape[0]

            if val_sample_count != y_val.shape[0]:
                raise IndexError('The validation inputs are not the same size as the labels \
                ({} and {}, respectively)'.format(val_sample_count,y_val.shape[0]))
            
            history['val_'+self.metric_name] = []

        if verbose:
            print('Training Network on {} sample{} for {} epoch{}...\
                '.format(sample_count, '' if sample_count == 1 else 's', epochs, '' if epochs == 1 else 's'))

        # Begins training
        for i in range(epochs):
            loss = 0

            # trains the network on the training data
            for j in range(sample_count):
                # Coerces 1D arrays into 2D for compatibility
                x_j = coerce_1d_array(X_train[j], axis=1)
                y_j = coerce_1d_array(y_train[j], axis=0)
                # forward propogation, using jth input
                output = self.forward_prop(x_j)
                # Calculates loss
                loss += self.loss_function(y_j, output)
                # backward propogation
                self.backward_prop(y_j, output)
                # Updates parameters
                self.update_params(epoch=i, time_step=j)
            
            # Gets metrics for history
            metric_output = self.evaluate(X_train, y_train)
            history[self.metric_name].append(metric_output)
            if validation_data is not None:
                metric_output = self.evaluate(X_val, y_val)
                history['val_'+self.metric_name].append(metric_output)
            
            # Gets average loss
            loss /= sample_count
            if verbose:
                print('Epoch {}/{}:  loss = {}'.format(i+1,epochs,loss))
        
        return history
    
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
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluates the model on testing data and labels
        
        :param X_test: The input for the testing data
        :param y_test: The labels for the testing data

        :returns: A value from the metric that was performed on the data
        """
        # Coerces 1D array into 2D array for compatibility
        Xt = coerce_1d_array(X_test, axis=1)
        yt = coerce_1d_array(y_test, axis=0)
        # Gets network prediction
        prediction = self.predict(Xt)
        # Returns the calculated metric score
        return self.metric(prediction, yt)

    def print_network(self) -> None:
        """Prints out a readble version of the network"""
        print('---------------------')
        print('NEURAL NETWORK PARAMS')
        print('---------------------\n')\
        
        if len(self.layers) == 0:
            print('Empty Network\n')
            return
        
        for l in range(len(self.layers)):
            layer = self.layers[l]
            print('Layer {}:'.format(l+1))
            print('---------')

            print('  ' + str(layer).replace('\n', '\n  ') + '\n')