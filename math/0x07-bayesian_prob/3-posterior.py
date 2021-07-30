#!/usr/bin/env python3
""" module to create posterior function """
import numpy as np


def intersection(x, n, P, Pr):
    """
    Function that calculates the intersection of obtaining this
    data given various hypothetical probabilities of developing
    severe side effects:
    Arguments:
     x is the number of patients that develop severe side effects
     n is the total number of patients observed
     P is a 1D numpy.ndarray containing the various hypothetical
       probabilities of developing severe side effects.
     Pr is a 1D numpy.ndarray containing the prior beliefs of P
    Returns:
    1D numpy.ndarray containing the intersection of obtaining
    x and n each probability in P, respectively
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if (type(P) is not np.ndarray) or (len(P.shape) != 1):
        raise TypeError("P must be a 1D numpy.ndarray")
    if (type(Pr) is not np.ndarray) or (P.shape != Pr.shape):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for i in range(len(P)):
        if i < 0 and i > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    for i in range(len(Pr)):
        if i < 0 and i > 1:
            raise ValueError("All values in Pr must be in the range [0, 1]")

    if np.isclose(np.sum(Pr), 1) is False:
        raise ValueError("Pr must sum to 1")

    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_n_x = np.math.factorial(n - x)
    factorial = (fact_n / (fact_x * fact_n_x))
    likelihood = factorial * (P ** x) * ((1 - P) ** (n - x))
    intersection = likelihood * Pr
    return intersection


def marginal(x, n, P, Pr):
    """ function which calculates the marginal probability of obtaining
    the data:
    Arguments:
    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects.
    Pr is a 1D numpy.ndarray containing the prior beliefs of P
    Returns:
    marginal probability of obtaining x and n
    """
    marginal = np.sum(intersection(x, n, P, Pr))
    return marginal


def posterior(x, n, P, Pr):
    """ function to calculate the posterior probability for the various
    hypothetical probabilities of developing severe side effects given
    the data:
    Returns:
    posterior probability of each probability in P given x and n, respectively
    """
    posterior = intersection(x, n, P, Pr) / marginal(x, n, P, Pr)
    return posterior
