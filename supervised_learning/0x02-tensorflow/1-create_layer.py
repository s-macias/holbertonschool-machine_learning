#!/usr/bin/env python3
""" Module to create the create_layer function """

import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

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
    layer = tf.layers.Dense(units=n,
    activation=activation,
    use_bias=True,
    kernel_initializer=init,
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name='layer')

    output = type(layer)

    return output