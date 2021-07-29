#!/usr/bin/env python3
""" module to create pca function """
import numpy as np


def pca(X, ndim):
    """ Function that performs PCA on a dataset

    Args:
        X (numpy.ndarray): array of shape (n, d) where
            n is the number of data points
            d is the number of dimensions of each point
        ndim : new dimensionality of the transformed X
    Returns:
    T (numpy.ndarray): array of shape (d, ndim) where nd is the new
    dimensionality of the transformed X.
    """
    X_centered = X - X.mean(axis=0)
    U, S, V = np.linalg.svd(X_centered)
    W = V[:ndim].T
    T = np.matmul(X_centered, W)
    return T
