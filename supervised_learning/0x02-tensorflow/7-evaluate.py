#!/usr/bin/env python3
""" Module to create evaluate function """


import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Function  that evaluates the output of a neural network:
    Arguments:
    X is a numpy.ndarray containing the input data to evaluate
    Y is a numpy.ndarray containing the one-hot labels for X
    save_path is the location to load the model from
    Import the meta graph
    Get the following from the graph’s collection:
    tensors y_pred, loss, and accuracy
    You are not allowed to use tf.saved_model
    Returns:
    the network’s prediction, accuracy, and loss, respectively
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss = sess.run(loss, feed_dict={x: X, y: Y})

        return (prediction, accuracy, loss)
