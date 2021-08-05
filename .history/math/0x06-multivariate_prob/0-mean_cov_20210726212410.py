#!/usr/bin/env python3
""" Module with functions to work with joint/multivariate distributions
""" import numpy as np


def mean_cov(X):
    """Function calculates the mean and covariance of a data set

    Args:
        X (numpy.ndarray): array of shape (n,d) containing the dataset, where:
            n (int): number of data points
            d (int): number of dimensions in each data point
    Returns:
        mean: numpy.ndarray of shape (1, d) containing the mean of the data set
        cov:  numpy.ndarray of shape (d, d) containing the covariance matrix of the data set
    """
    if type(X) != np.ndarray or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    
    if X.shape[0] < 2:
        