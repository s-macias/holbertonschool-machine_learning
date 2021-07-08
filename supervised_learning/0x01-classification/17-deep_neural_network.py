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
