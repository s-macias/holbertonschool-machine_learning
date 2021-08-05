#!/usr/bin/env python3
    """ 
    Module with functions to work with joint/multivariate distributions
    """
    
def mean_cov(X):
    """Function calculates the mean and covariance of a data set

    Args:
        X (numpy.ndarray): array of shape (n,d) containing the dataset, where:
            n (int): number of data points
            d (int): number of dimensions in each data point
            
    """
    