#!/usr/bin/env python3
""" Updates Exponential class """


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

    def pdf(self, x):
        """
        Instance method to calculate the pdf of a given time period
        x is the time period
        Return: PDF value for x or 0 if x is out of range
        """
        if x < 0:
            return 0

        e = 2.7182818285
        pdf = self.lambtha * pow(e, -self.lambtha * x)

        return pdf

    def cdf(self, x):
        """
        Instance method to calculate the CDF for a given time period
        x is the time period
        Returns the CDF value for x or 0 if x is out of range
        """
        if x < 0:
            return 0

        e = 2.7182818285
        cdf = 1 - pow(e, -self.lambtha * x)

        return cdf
