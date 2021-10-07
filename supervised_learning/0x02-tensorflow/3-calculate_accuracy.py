#!/usr/bin/env python3
""" Module with calculate_accuracy function """

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Function that calculates the accuracy of a prediction
    Arguments:
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns:
    a tensor containing the decimal accuracy of the prediction
    hint: accuracy = correct_predictions / all_predictions
    """
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    return tf.reduce_mean(tf.cast(correct_predictions, 'float32'))
