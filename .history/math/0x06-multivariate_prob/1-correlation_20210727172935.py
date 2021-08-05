#!/usr/bin/e
    """ 
    Module with functions to work with joint/multivariate distributions
    """
    import numpy as np
    
def correlation(C):
    """
    function that calculates a correlation matrix

    Args:
        C ([numpyarray]):  shape (d, d) containing a covariance matrix
            
    """