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
        raise TypeError("C must be a numpy.ndarray")
    if C.shape[0] != C.shape[1] or len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")
    
    corr_mat = C
    
    return corr_mat
    