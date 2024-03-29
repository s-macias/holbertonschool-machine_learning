#!/usr/bin/env python3
""" Updates Binomial class """


class Exponential():
    """ Exponential Distribution """

    def __init__(self, data=None, lambtha=1.):
        """
         initializing Exponential class
        data: list with data to estimate the the distribution
        lambtha is the expected number of occurrences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = len(data) / sum(data)
