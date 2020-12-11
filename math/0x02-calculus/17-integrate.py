#!/usr/bin/env python3
""" calculates the integral  of a polynomial """


def poly_integral(poly, C=0):
    """ calculates the integral of a polynomial """
    if type(poly) is not list or poly is None or type(C) is not int:
        return None
    integral = []
    i = 0
    for coef in poly:
        item = poly[i] / (i + 1)
        if type(item) is not float:
            item = int(item)
        integral.append(item)
        i += 1
    return integral
