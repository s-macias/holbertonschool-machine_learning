#!/usr/bin/env python3
""" adds two matrices element-wise """


def add_matrices2D(mat1, mat2):
    """ adds two matrices of ints/floatselement-wise """
    matrix_shape = __import__('2-size_me_please').matrix_shape
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    else:
        new_array = []
        for row in range(len(mat1)):
            columns = []
            for column in range(len(mat1[row])):
                columns.append(mat1[row][column] + mat2[row][column])
            new_array.append(columns)
        return new_array
