#!/usr/bin/env python3
""" module to create pca function """
import numpy as np


def pca(X, var=0.95):
    """ Function that performs PCA on a dataset

    Args:
        X (numpy.ndarray): array of shape (,d) where
            n is the number of data points
            d is the number of dimensions of each point
        var (float, optional): fraction of the variance that the PCA
        transformation should maintain. Defaults to 0.95.
    Returns:
    
    """