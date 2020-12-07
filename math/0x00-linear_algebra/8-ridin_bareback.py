#!/usr/bin/env python3
""" Performs matrix multiplication """


def mat_mul(mat1, mat2):
    """ Performs matrix multiplication of two 2D matrices """
    matrix_shape = __import__('2-size_me_please').matrix_shape
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    else:
        new_array = []
        for row in range(len(mat1)):
            temp = []
            for column in range(len(mat2[0])):
                position = 0
                for count in range(len(mat1[0])):
                    element = mat1[row][count] * mat2[count][column]
                    position += element
                row.append(position)
            new_array.append(temp)
        return new_array
