#!/usr/bin/env python3
""" adds two matrices """
import numpy as np


def add_matrices(mat1, mat2):
    """ adds two matrices """
    if np.shape(mat1) != np.shape(mat2):
        return None
    else:
        addition = np.add(mat1, mat2)
        return addition
