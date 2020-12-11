#!/usr/bin/env python3
""" calculates a summation of i squared """


def summation_i_squared(n):
    """ calculates a summation of i squared """
    if type(n) is not int or n < 1:
        return None
    else:
        return int(n*(n+1)*((2*n)+1)/6)
