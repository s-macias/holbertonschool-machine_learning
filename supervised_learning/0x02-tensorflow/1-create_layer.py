#!/usr/bin/env python3
""" Module to create the create_layer function """

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Function to create a layer in a neural network
    Arguments:
        prev is the tensor output of the previous layer
        n is the number of nodes in the layer to create
        activation is the activation function that the layer should use
    use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    to implement He et. al initialization for the layer weights
    each layer should be given the name layer
    Returns:
    the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation, use_bias=True,
                            kernel_initializer=init, name='layer')

    return layer(prev)
