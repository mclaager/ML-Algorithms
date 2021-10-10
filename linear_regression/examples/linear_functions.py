import numpy as np

from linear_regression.linear_regression import LinearRegression

from utils.helper_functions import coerce_1d_array
from utils.init_methods import randn

class linear_function():
    """A class that implements linear functions"""
    def __init__(self, w_and_b: np.ndarray):
        """
        Saves bias and weights as the function

        :param w_and_b: A (1,1,1+|w|) dimensional array, where the 3rd dimension is the bias and weights
        """
        self.func = w_and_b
    
    def output(self, x: np.ndarray) -> np.ndarray:
        """
        Takes an input array of weights and applies the function to it, adding bias automatically.

        :param x: An array of shape (m,n), where m is the number of samples and n is the length
        of the input vector
        :returns: An array of output(s) from the linear function
        """
        # Compatibility...
        X_new = coerce_1d_array(x, axis=1)
        # Creates the bias array
        bias = np.ones((X_new.shape[0],))
        # Creates an (m,n+1) dimension array, with the bias terms leading
        X_new = np.column_stack((bias,X_new))
        return X_new @ self.func


# Function that is implemented... y= 3 + 2(x1) + 5(x2)
# ---CHANGE THESE TERMS TO TRAIN ON DIFFERENT LINEAR DATA---
BIAS_TERM = [3]
WEIGHT_TERMS = [2,5]
# Create the linear function
lf = linear_function(np.concatenate((BIAS_TERM, WEIGHT_TERMS)))
func = lf.output

# Prints original function
orig_str = 'Original Function:\n\
    y = {0:.3f}'.format(BIAS_TERM[0])
for i in range(len(WEIGHT_TERMS)):
    orig_str += ' + {0:.3f}*(x{1})'.format(WEIGHT_TERMS[i],i+1)
print(orig_str)


# Number of datapoints to train on
m = 200

# Creates training data that has normal noise added
x_n = [2 * randn((m,1)) for i in range(len(WEIGHT_TERMS))]
X_train = np.column_stack(x_n)
y_train = func(X_train) + randn((m,))


# Performs closed form linear regression
closed_model = LinearRegression()
closed_model.fit_closed(X_train, y_train)

# Prints least squares prediction
closed_form_str = 'Closed-Form Linear Regression (Least-Squares) Prediction:\n\
    y = {0:.3f}'.format(closed_model.get_bias())
closed_form_weights = closed_model.get_weights()
for i in range(closed_form_weights.shape[0]):
    closed_form_str += ' + {0:.3f}*(x{1})'.format(closed_form_weights[i],i+1)
print(closed_form_str)


# Performs mini-batch SGD linear regression
epochs = 100
lr = 0.01
batch_size = 4
sgd_model = LinearRegression(optimizer='SGD')
sgd_model.fit(X_train, y_train, epochs=epochs, lr=lr, batch_size=batch_size)

# Prints SGD prediction
sgd_str = 'Mini-batch Gradient Descent Prediction (epochs={0}, learning rate={1}, batch size={2}):\n\
    y = {3:.3f}'.format(epochs, lr, batch_size, sgd_model.get_bias())
sgd_weights = sgd_model.get_weights()
for i in range(sgd_weights.shape[0]):
    sgd_str += ' + {0:.3f}*(x{1})'.format(sgd_weights[i],i+1)
print(sgd_str)


# Prints original function output on a set of weights [1, 2, .., n]
one_arr = [[i+1 for i in range(len(WEIGHT_TERMS))]]
print('\nOriginal function output on input [1, 2, ..., n-1, n]:\n\
    {}'.format(func(one_arr)))
# Prints closed form prediction on same set of weights
print('Least squares prediction on input [1, 2, ..., n-1, n]:\n\
    {}'.format(closed_model.predict(one_arr)))
# Prints SGD prediction on same set of weights
print('SGD prediction on input [1, 2, ..., n-1, n]:\n\
    {}'.format(sgd_model.predict(one_arr)))