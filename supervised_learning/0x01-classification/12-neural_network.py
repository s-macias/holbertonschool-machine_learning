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

        Private instance attributes:
        W1: The weights vector for hidden layer
        Instance should be initialized using a random normal dist.
        b1: The bias for the hidden layer.
        Instance should be initialized to 0.
        A1: The activated output of hidden layer
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

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ W1 getter function """
        return self.__W1

    @property
    def b1(self):
        """ b1 getter function """
        return self.__b1

    @property
    def A1(self):
        """ A1 getter function """
        return self.__A1

    @property
    def W2(self):
        """ W2 getter function """
        return self.__W2

    @property
    def b2(self):
        """ b2 getter function """
        return self.__b2

    @property
    def A2(self):
        """ A2 getter function """
        return self.__A2

    def forward_prop(self, X):
        """
        Public method to calculate the forward propagation of the
        neural network
        Arguments:
        X: numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        Returns:
        private attributes __A1 and __A2, respectively
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Public method to calculates the cost of the model using
        logistic regression
        Arguments:
        Y is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        A is a numpy.ndarray with shape (1, m) containing the
        activated output of the neuron for each example
        To avoid division by zero errors, please use 1.0000001 - A
        instead of 1 - A
        Returns:
        cost
        """
        c = - (1 * np.sum(Y * np.log(A) + (1 - Y) *
               np.log(1.0000001 - A))) / (Y.shape[1])
        return c

    def evaluate(self, X, Y):
        """
        Public method to evaluate the neural network's predictions
        Arguments:
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        Y is a numpy.ndarray with shape (1, m) that contains the correct labels
        for the input data
        Returns:
        neuron’s prediction and the cost of the network, respectively
        The prediction should be a numpy.ndarray with shape (1, m) containing
        the predicted labels for each example
        The label values should be 1 if the output of the network is >= 0.5
        and 0 otherwise
        """
        A1, A2 = self.forward_prop(X)
        a = np.where(A2 < 0.5, 0, 1)
        return a, self.cost(Y, A2)
