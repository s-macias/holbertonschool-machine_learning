#!/usr/bin/env python3
""" Module to create class GaussianProcess """
import numpy as np


class GaussianProcess():
    """
    class that represents a noiseless 1D Gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        class constructor
        Arguments:
        X_init is a numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing the outputs of
        the black-box function for each input in X_init
        t is the number of initial samples
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of the black-box
        function
        Public instance attributes:
        Sets the public instance attributes X, Y, l, and sigma_f corresponding
        to the respective constructor inputs
        Sets the public instance attribute K, representing the current
        covariance kernel matrix for the Gaussian process
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, Y_init)

    def kernel(self, X1, X2):
        """
        Function that calculates the covariance kernel matrix between two
        matrices: X1 and X2
        X1 is a numpy.ndarray of shape (m, 1)
        X2 is a numpy.ndarray of shape (n, 1)
        the kernel should use the Radial Basis Function (RBF)
        Returns:
        covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(
            X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-.5 * (1/self.l ** 2) * sqdist)