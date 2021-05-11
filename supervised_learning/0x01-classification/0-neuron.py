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
        W is a public instance attribute. The weights vector for the neuron.
        Instance should be initialized using a random normal dist.
        b is a public instance attribute. The bias for the neuron.
        Instance should be initialized to 0.
        A is a public instance attribute. The activated output of the neuron.
        Instance should be initialized to 0.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
