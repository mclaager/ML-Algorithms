import numpy as np

import utils.loss_functions as loss_functions
import utils.init_methods as init_methods
import utils.optimizers as optimizers
from utils.helper_functions import coerce_1d_array, str_or_class, str_or_func

class LinearRegression():
    def __init__(self, beta_init_method: str or function = 'randu', optimizer = 'SGD') -> None:
        """Initializes methods that will be used in fitting the model"""
        self.beta_init_method = str_or_func(init_methods, beta_init_method,\
            'Invalid beta initialization function given.')
        self.optimizer = str_or_class(optimizers, optimizer,\
            'Invalid optimizer given.')
        self.beta = None
        self.deltas = {}
        pass

    # Uses the closed form of linear regression (least squares) to get weights
    def fit_closed(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """
        Uses the closed form of linear regression (least-squares) to make a prediction of
        the weights and bias of the function

        :param X_train: The training data, sample dimension must come first
        :param y_train: The training output, sample dimension must come first

        :returns: Predicted beta, where beta is a vector with the pred. bias + pred. weights
        """
        # Compatibility...
        X_t = coerce_1d_array(X_train, axis=0)
        y_t = coerce_1d_array(y_train, axis=0)

        # Column stacks x1 ... xn to form a matrix X of the bias terms
        # (intialized to 1) and x_i.
        x_new = [init_methods.ones((X_t.shape[0],1))] + [X_t]
        x_new = np.column_stack(x_new)

        # Least squares
        beta = np.linalg.inv(x_new.T @ x_new) @ x_new.T @ y_t
        beta = beta.flatten()

        # save weights and bias
        self.beta = beta
        return beta

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, lr: float, batch_size: int = 1) -> dict:
        """Uses optimization, rather than the closed form solution, to fit the training data."""
        # Compatibility...
        X_t = coerce_1d_array(X_train, axis=0)
        y_t = coerce_1d_array(y_train, axis=0)

        # Gets number of training samples
        sample_count = X_t.shape[0]

        # Column stacks x1 ... xn to form a matrix X of the bias terms
        # (intialized to 1) and x_i.
        x_new = [init_methods.ones((sample_count,1))] + [X_t]
        x_new = np.column_stack(x_new)

        # Generates an initial guess for beta
        self.beta = self.beta_init_method((x_new.shape[1],1))

        # Initializes history
        history = {
            'betas' : [self.beta.flatten()]
        }
        # Sets the learning rate of the optimizer
        self.optimizer.set_lr(lr)

        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(sample_count)
            xi_shuffled = x_new[shuffled_indices]
            y_shuffled = y_t[shuffled_indices]
            # Runs each mini-batch
            for i in range(0, sample_count, batch_size):
                # Determines if there is cutoff for the batch
                end_indx = min(i+batch_size, sample_count)
                # Creates training batch
                xi = xi_shuffled[i:end_indx]
                yi = y_shuffled[i:end_indx]
                # Calculates gradient (the error of the beta term)
                self.deltas['beta'] = 1. / (end_indx-i) * (xi.T @ (xi @ self.beta - yi))
                # Optimizes beta term and adds it to history
                self.optimizer.optimize(trainable=self, batch_size=batch_size, epoch=epoch, time_step=i)
                history['betas'].append(self.beta.flatten())

        # Flattens the beta term
        self.beta = self.beta.flatten()
        return history

    def predict(self, input):
        # Compatibility
        x = coerce_1d_array(input, axis=0)
        # Adds bias term
        x_new = [init_methods.ones((x.shape[0],1))] + [x]
        x_new = np.column_stack(x_new)
        return x_new @ self.beta

    def get_bias(self) -> float:
        return self.beta[0]
    
    def get_weights(self) -> np.ndarray:
        return self.beta[1:]
    
    def get_beta(self) -> np.ndarray:
        return self.beta
    
    def get_deltas(self) -> dict:
        return self.deltas
