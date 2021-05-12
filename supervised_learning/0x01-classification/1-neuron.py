#!/usr/bin/env python3
""" write a class Neuron """
import numpy as np


class Neuron:
    """
    class Neuron that defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        """ Class constructor
        Arguments:
        nx is the number of input features to the neuron
        Public instance attributes:
        W: The weights vector for the neuron.
           Instance should be initialized using a random normal dist.
        b:  The bias for the neuron.
            Instance should be initialized to 0.
        A:  The activated output of the neuron.
            Instance should be initialized to 0.
        Private instance attributes:
        __W: The weights vector for the neuron.
             Instance should be initialized using a random normal dist.
        __b: The bias for the neuron.
             Instance should be initialized to 0.
        __A: The activated output of the neuron (prediction).
             Instance should be initialized to 0.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

        @property
        def W(self):
            """ W (weight) getter function """
            return self.__W

        @property
        def b(self):
            """ b (bias) getter function """
            return self.__b

        @property
        def A(self):
            """ A (Activation values) getter function """
            return self.__A
