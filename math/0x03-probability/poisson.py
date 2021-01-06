#!/usr/bin/env python3
""" Creates a class Poisson that represents a poisson distribution """


class Poisson:
    """ class that represents a Poisson distribution """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor that initializes the class Poisson
        Parameters:
        data:  list of the data used to estimate the distribution
        lambtha: expected number of occurences in a given time frame
        Return:
        the lambtha of data if data is given
        Errors:
        ValueError: raised If lambtha is not a positive value or equals to 0
        TypeError: raised If data is not a list
        """
        if data is None and lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        elif data is None and lambtha > 0:
            self.lambtha = float(lambtha)
        elif data and type(data) is not list:
            raise TypeError("data must be a list")
        elif data and len(data) < 2:
            raise ValueError("data must contain multiple values")
        else:
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        method that calculates the PMF for a given number of successes
        Parameters:
        k: number of successes
        Return:
        pmf value for k
        """
        if k <= 0:
            return 0

        k = int(k)
        e = 2.7182818285

        k_factorial = 1
        for i in range(1, k + 1):
            k_factorial *= i

        pmf = pow(e, -self.lambtha) * pow(self.lambtha, k) / k_factorial
        return pmf

    def cdf(self, k):
        """ method that calculates the CDF value for a # of successes
        Parameters:
        k: number of successes
        Return:
        cdf value for k
        """
        if k <= 0:
            return 0

        k = int(k)
        e = 2.7182818285
        cdf = 0
        for i in range(1, k + 1):
            cdf += self.pmf(i)
        return cdf
