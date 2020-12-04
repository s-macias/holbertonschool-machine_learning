#!/usr/bin/env python3
""" function that calculates the shape of a matrix"""
import numpy as np


def matrix_shape(matrix):
    """ calculates the shape of a matrix: [# of rows, # of columns]"""
    shape = []
    shape.append(len(matrix[0]))
    shape.append(len(matrix))
    return shape
