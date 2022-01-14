"""
Functions that are able to be used to get metrics on training/testing data.
"""

import numpy as np

def binary_accuracy(output, labels):
    """
    Gets the accuracy of a binary classification model based on its prediction.
    Accepts float values [0,1]
    
    :param output: The predictions of the model
    :param labels: The true labels for the data

    :returns: The accuracy of the model as a float in range [0,1]
    """
    # Rounds predictions to be either 0 or 1. np.rint causes the labels to be
    # discrete predictions; 0 on [0, 0.5] and 1 on (0.5, 1].
    prediction = np.rint(output)
    # Returns the accuracy of the network on the testing set
    return np.sum(prediction == labels) / prediction.size

def class_accuracy(prediction, true_labels):
    """
    Gets the accuracy of multi-class classification model based on its prediction
    
    :param prediction: The predictions of the model
    :param true_labels: The true labels for the data

    :returns: The accuracy of the model as a float in range [0,1]
    """
    return np.sum(prediction == true_labels) / prediction.size