# Machine Learning Algorithms
This repository contains all of my custom implementations of various different machine learning algorithms. It serves as a learning opportunity and a way to express these concepts as modularized, easy-to-understand code.


## Included Algorithms

### 1. Neural Network

Dependencies: NumPy

An implementation of a fully-connected deep neural network that is styled after Keras.

It currently allows for multi-layered networks to be easily created and trained. Features dense layers (fully-connected layers) and activation layers. Has built-in activation functions, weight+bias initialization methods, cost functions, and metrics; however, custom functions are easily able to be put in place of the built-in methods. Currently only uses pure SGD (not mini-batch) optimizer.

Examples can be seen under `neural_network/examples`.

### 2. Linear Regression

Dependencies: NumPy

Implements the closed-form solution for performing linear regression. Also shows how **Polynomial Regression** can be implemented as an expansion of linear regression under `linear_regression/examples/polynomial_regression.ipynb`.


## Planned Additions

The following are considerations for algorithms that may be implemented here. Some of these algorithms exist in my other existing repositories, but I still need to standarize them or add additional functionality to make them more generalized. None are guaranteed to be added.

* Different Optimizers for all ML algorithms, which will be available as a collection within `tools/`
* Logistic regression
* Optimizer-based linear regression
* Statistic analysis functions for models ($r^2$ score, AIC, BIC, etc.)
* Convolution Neural Netorks
* Support Vector Machines
* Random Forest
* K Nearest Neighbors
* Principal Component Analysis
* Fischer Linear Discriminant Analysis
* RL Algorithms (TBD)