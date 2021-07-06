#!/usr/bin/env python3
""" write a class Neural network """
import numpy as np


class NeuralNetwork:
    """
    class NeuralNetwork that defines a neural network with one hidden layer
    performing binary classification
    """
    def __init__(self, nx, nodes):
        """ Class constructor
        Arguments:
        nx is the number of input features to the neural network
        W1 is a public instance attribute. The weights vector for hidden layer
        Instance should be initialized using a random normal dist.
        b1 is a public instance attribute. The bias for the hidden layer.
        Instance should be initialized to 0.
        A1 is a public instance attribute. The activated output of hidden layer
        Instance should be initialized to 0.
        W2: The weights vector for the output neuron. Upon instantiation,
        it should be initialized using a random normal distribution.
        b2: The bias for the output neuron. Upon instantiation,
        it should be initialized to 0.
        A2: The activated output for the output neuron (prediction).
        Upon instantiation, it should be initialized to 0.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
