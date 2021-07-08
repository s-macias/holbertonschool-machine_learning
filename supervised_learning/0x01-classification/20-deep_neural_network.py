#!/usr/bin/env python3
""" write a class DeepNeuralNetwork class """
import numpy as np


class DeepNeuralNetwork:
    """
    class DeepNeuralNetwork that defines a neural network
    performing binary classification
    """
    def __init__(self, nx, layers):
        """ Class constructor
        Arguments:
        nx [int] is the number of input features to the neural network
        layers [list] is a list representing the number of nodes in each layer
        Public attributes:
        L: The number of layers in the neural network.
        cache: A dictionary to hold all intermediary values of the network.
        Upon instantiation, it should be set to an empty dictionary.
        weights: A dictionary to hold all weights and biased of the network.
        The weights of the network should be initialized using the He et al.
        method and saved in the weights dictionary using the key W{l} where {l}
        is the hidden layer the weight belongs to
        The biases of the network should be initialized to 0’s and saved in the
        weights dictionary using the key b{l} where {l} is the hidden layer the
        bias belongs to
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__nx = nx
        self.__layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer in range(len(layers)):
            if type(layers[layer]) is not int or layers[layer] <= 0:
                raise TypeError("layers must be a list of positive integers")
            else:
                self.__weights[
                    "b" + str(layer + 1)] = np.zeros((layers[layer], 1))
                if layer == 0:
                    self.__weights[
                        "W" + str(layer + 1)] = np.random.randn(
                            layers[layer], nx) * np.sqrt(2/nx)
                else:
                    self.__weights[
                        "W" + str(layer + 1)] = np.random.randn(
                            layers[layer], layers[layer - 1]) * np.sqrt(
                                2 / layers[layer - 1])

    @property
    def L(self):
        """
        L getter function
        Returns:
        Number of layers
        """
        return self.__L

    @property
    def cache(self):
        """
        cache getter function
        Returns:
        Dictionary with intermediary values of the network
        """
        return self.__cache

    @property
    def weights(self):
        """
        weights getter function
        Returns:
        Dictionary with weights and bias values of the network
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Public method to calculate the forward propagation of the
        neural network
        Arguments:
        X: numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        Updates the private attribute __cache
        The activated outputs of each layer should be saved in the __cache
        dictionary using the key A{l} where {l} is the hidden layer the
        activated output belongs to
        X should be saved to the cache dictionary using the key A0
        All neurons should use a sigmoid activation function
        Returns:
        output of the neural network and the cache, respectively
        """
        self.__cache['A0'] = X
        for layer in range(self.__L):
            # z = W(l + 1) * A(l)  + b(l + 1)
            z = np.matmul(self.__weights["W" + str(layer + 1)], self.__cache[
                "A" + str(layer)]) + self.__weights["b" + str(layer + 1)]
            A_value = 1 / (1 + np.exp(-z))
            self.__cache["A" + str(layer + 1)] = A_value

        return self.__cache["A" + str(layer + 1)], self.__cache

    def cost(self, Y, A):
        """
        Public method to calculates the cost of the model using
        logistic regression
        Arguments:
        Y is a numpy.ndarray with shape (1, m) that contains the correct labels
            for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated output
            of the neuron for each example
        To avoid division by zero errors,
        please use 1.0000001 - A instead of 1 - A
        Returns:
        cost
        """
        cost = - (1 * np.sum(Y * np.log(A) + (1 - Y) * np.log(
            1.0000001 - A))) / (Y.shape[1])
        return cost

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
        The label values should be 1 if the output of the network is >= 0.5 and
        0 otherwise
        """
        A, _ = self.forward_prop(X)
        output = np.where(A < 0.5, 0, 1)
        return output, self.cost(Y, A)
