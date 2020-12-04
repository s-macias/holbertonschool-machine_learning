#!/usr/bin/env python3
""" adds two matrices element-wise """


def add_matrices2D(mat1, mat2):
    """ adds two matrices of ints/floatselement-wise """
    matrix_shape = __import__('2-size_me_please').matrix_shape
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    else:
        new_array = []
        for item in range(len(mat1)):
            column = []
            for col in range(len(mat1[0])):
                column.append(mat1[item][col] + mat2[item][col])
            new_array.append(column)
        return new_array
