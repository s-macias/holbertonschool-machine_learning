#!/usr/bin/env python3
""" module to create markov_chain function """

import numpy as np


def markov_chain(P, s, t=1):
    """
    Function that determines the probability of a markov chain being in a
    particular state after a specified number of iterations:
    Arguments:
    P is a square 2D numpy.ndarray of shape (n, n) representing the transition
    matrix
    P[i, j] is the probability of transitioning from state i to state j
    n is the number of states in the markov chain
    s is a numpy.ndarray of shape (1, n) representing the probability of
    starting in each state
    t is the number of iterations that the markov chain has been through
    Returns
    numpy.ndarray of shape (1, n) representing the probability of being in a
    specific state after t iterations, or None on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if type(s) is not np.ndarray:
        return None
    if s.shape[1] != P.shape[0]:
        return None
    if s.shape[0] != 1:
        return None
    if type(t) is not int or t < 0:
        return None

    for i in range(t):
        s = np.matmul(s, P)
    return s
