#!/usr/bin/env python3
""" Module to create create_train_op function """

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Function that creates the training operation for the network
    Arguments:
    loss is the loss of the network’s predictions
    alpha is the learning rate
    Returns:
    an operation that trains the network using gradient descent
    """
    operation = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return operation
