#!/usr/bin/env python3
""" performs elementwise operations on matrices """


def np_elementwise(mat1, mat2):
    """ performs element-wise add, subtrac, multiply, and divide """
    addition = mat1 + mat2
    substraction = mat1 - mat2
    multiplication = mat1 * mat2
    division = mat1 / mat2
    return (addition, substraction, multiplication, division)
