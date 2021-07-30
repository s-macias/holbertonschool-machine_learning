#!/usr/bin/env python3
""" module to create likehood function """
import numpy as np


def likelihood(x, n, P):
    """
    Function that calculates the likelihood of obtaining this
    data given various hypothetical probabilities of developing
    severe side effects:
    Arguments:
     x is the number of patients that develop severe side effects
     n is the total number of patients observed
     P is a 1D numpy.ndarray containing the various hypothetical
       probabilities of developing severe side effects.
    Returns:
    1D numpy.ndarray containing the likelihood of obtaining the data,
    x and n, for each probability in P, respectively
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for i in range(len(P)):
        if i < 0 and i > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    # binomial distribution:
    # P(x:n,p) = n!/[x!(n-x)!].px.(q)n-x
    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_n_x = np.math.factorial(n - x)
    factorial = (fact_n / (fact_x * fact_n_x))
    likelihood = factorial * (P ** x) * ((1 - P) ** (n - x))
    return likelihood
