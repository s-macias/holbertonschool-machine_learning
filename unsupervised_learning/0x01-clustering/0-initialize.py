#!/usr/bin/env python3
""" module to initialize centroids """

import numpy as np


def initialize(X, k):
    """
    Function that hat initializes cluster centroids for K-means
    Args:
    X is a numpy.ndarray of shape (n, d) containing the dataset
        that will be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    The cluster centroids should be initialized with a multivariate uniform
    distribution along each dimension in d:
    The minimum values for the distribution should be the minimum values of X
    along each dimension in d
    The maximum values for the distribution should be the maximum values of X
    along each dimension in d
    You should use numpy.random.uniform exactly once
    Returns:
    numpy.ndarray of shape (k, d) containing the initialized centroids for each
    cluster, or None on failure
    """
    if type(X) is not np.ndarray:
        return None
    if len(X.shape) != 2:
        return None
    if type(k) is not int or k < 0:
        return None

    n, d = X.shape

    centroids = np.random.uniform(X.min(axis=0), X.max(axis=0), (k, d))

    return centroids
