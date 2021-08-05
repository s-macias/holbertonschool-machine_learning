#!/usr/bin/e
    """ 
    Module with functions to work with joint/multivariate distributions
    """
    import numpy as np
    
def correlation(C):
    """
    function that calculates a correlation matrix

    Args:
        C ([numpy.ndarray]):  shape (d, d) containing a covariance matrix
            d is the number of dimensions 
    Returns:
    numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if type(C) is not numpy.ndarray:
        raise TypeError("")