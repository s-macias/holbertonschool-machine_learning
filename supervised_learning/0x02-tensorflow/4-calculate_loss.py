#!/usr/bin/env python3
""" module to create calculate_loss function """

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Function that calculates the softmax
    cross-entropy loss of a prediction
    Arguments:
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns:
    a tensor containing the loss of the prediction
    """
    cross_entropy = tf.losses.softmax_cross_entropy(y, y_pred)
    return cross_entropy
