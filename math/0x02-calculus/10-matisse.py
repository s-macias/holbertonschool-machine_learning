#!/usr/bin/ env python3
""" calculates the derivative of a polynomial """


def poly_derivative(poly):
    """ calculates the derivative of a polynomial """
    if type(poly) is not list or poly is None:
        return None
    elif len(poly) == 1:
        return [0]
    derivative = []
    i = 0
    for coef in poly:
        item = i * poly[i]
        derivative.append(item)
        i += 1
    return derivative[1:]
