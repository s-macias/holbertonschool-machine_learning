#!/usr/bin/env python3
""" function that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """ calculates the shape of a matrix: [# of rows, # of columns]"""
    shape = []
    if type(matrix) == list:
        shape.append(len(matrix))
        shape = shape + matrix_shape(matrix[0])
    return shape
