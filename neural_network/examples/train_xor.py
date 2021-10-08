import numpy as np

from neural_network.layers.activation_layer import ActivationLayer
from neural_network.layers.dense_layer import DenseLayer

from neural_network.network import NeuralNetwork

# Define dimensions for neural network
input_dims = 2 # [0,1], [1,0], etc.
hidden_dims = 3
output_dims = 1

# Create layers
layer1 = DenseLayer(input_dims,hidden_dims)
layer1_a = ActivationLayer('tanh')
layer2 = DenseLayer(hidden_dims, output_dims)
layer2_a = ActivationLayer('tanh')

# Create network
network = NeuralNetwork([
    layer1,
    layer1_a,
    layer2,
    layer2_a
])

# training data
X_train = np.vstack(([0,0],[0,1],[1,0],[1,1]))
y_train = np.array([0,1,1,0]).reshape(-1,1)

# Compliles the network (uses sgd optimizer, mse loss function, binary accuracy metric)
network.compile()

# Fit the network
network.fit(X_train=X_train, y_train=y_train, epochs=1000, lr=0.1)

# Prove that network was able to fit data well
yHat = network.predict(np.array([[0,0]]))
print('\nPrediction on [0,0]: {}'.format(yHat))
yHat = network.predict(np.array([[0,1]]))
print('Prediction on [0,1]: {}'.format(yHat))
yHat = network.predict(np.array([[1,0]]))
print('Prediction on [1,0]: {}'.format(yHat))
yHat = network.predict(np.array([[1,1]]))
print('Prediction on [1,1]: {}\n'.format(yHat))

# Print the weights of the network
network.print_network()