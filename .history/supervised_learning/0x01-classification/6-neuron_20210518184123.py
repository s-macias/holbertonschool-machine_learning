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

    def forward_prop(self, X):
        """
        Public method to calculate forward propagation of the neuron.
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Updates the private attribute __A (after sigmoid has been applied)
        The neuron should use a sigmoid activation function.
        Returns: private attribute __A
        """
        self.__A = 1 / (1 + np.exp(-1 * (np.dot(self.__W, X) + self.__b)))
        return self.__A

    def cost(self, Y, A):
        """
        Public method to calculate the cost of the model using logistic
        regression.
        Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        To avoid division by zero errors, please use 1.0000001 - A instead
        of 1 - A
        Returns the cost
        """
        c = - (1 * np.sum(Y * np.log(A) + (1 - Y) *
                          np.log(1.0000001 - A))) / (Y.shape[1])

        return c
    
    def evaluate(self, X, Y):
        """
        Public method to evaluate the neuron’s predictions
        Arguments:
        X is a numpy.ndarray with shape (nx, m) that contains
        the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Y is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        Returns the neuron’s prediction and the cost of the network,
        respectively
        The prediction should be a numpy.ndarray with shape (1, m)
        containing the predicted labels for each example
        The label values should be 1 if the output of the network
        is >= 0.5 and 0 otherwise
        """
        A = np.self.forward_prop(x)
        a = np.where(A < 0.5, 0, 1)
        return A, self.cost(Y, A)
    
    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Public method to calculate one pass of gradient
        descent on the neuron

        Args:
            X ([numpy.ndarray]): [shape (nx, m) that
            contains the input data
            nx is the number of input features to the neuron
            m is the number of examples]
            Y ([numpy.ndarray ]): [shape (1, m) that contains
            the correct labels for the input data]
            A ([numpy.ndarray]): [shape (1, m) containing
            the activated output of the neuron for each example]
            alpha (float, optional): [ learning rate]. Defaults to 0.05.
            Updates the private attributes __W and __b
        """
        m = Y.shape[1]
        dw = np.matmul(A - Y, X.T) / m
        db = np.sum(A - Y) / m
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)
        
    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Public method to calculate one pass of gradient
        descent on the neuron

        Args:
            X ([numpy.ndarray]): [shape (nx, m) that
            Trains the neuron
            nx is the number of input features to the neuron
            m is the number of examples]
            Y ([numpy.ndarray]): [shape (1, m) that contains
            the correct labels for the input data]
            iterations ([int]): number of iterations to train over
            if iterations is not an integer, raise a TypeError with
            the exception iterations must be an integer
            if iterations is not positive, raise a ValueError with
            the exception iterations must be a positive integer
            alpha (float, optional): [ learning rate]. Defaults to 0.05.
            if alpha is not a float, raise a TypeError with the
            exception alpha must be a float
            if alpha is not positive, raise a ValueError with
            the exception alpha must be positive
            Updates the private attributes __W, __b, and __A
            Returns the evaluation of the training data after
            iterations of training have occurred
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        m = Y.shape[1]
        Cost = []
        Iteration = []
        for i 
    
       
        
           
        
        