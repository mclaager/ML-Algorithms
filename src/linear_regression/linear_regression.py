import numpy as np

import ml_functions.cost_functions as cost_functions
import ml_functions.init_methods as init_methods
from ml_functions.helper_functions import coerce_1d_array, str_or_func

class LinearRegression():
    def __init__(self, weight_init_method: str or function = 'randu',\
        bias_init_method: str or function = 'randu', cost_function: str or function = 'mse') -> None:
        self.weight_init_method = str_or_func(weight_init_method, init_methods,\
            'Invalid weight initialization function given.')
        self.bias_init_method = str_or_func(bias_init_method, init_methods,\
            'Invalid bias initialization function given.')
        self.loss_function = str_or_func(cost_function, cost_functions,\
            'Invalid loss function given.')
        self.beta = None
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

    def fit(self, X_train, y_train, epochs, lr) -> list:
        """
        Uses another method other than the closed form to fit the training data.
        """
        raise NotImplementedError

    def predict(self, input):
        # Compatibility
        x = coerce_1d_array(input, axis=1)
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
