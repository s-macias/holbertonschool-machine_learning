#!/usr/bin/env python3
""" concatenates two matrices along a specific axis """


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenates two matrices along a specific axis """
    concat = []
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        else:
            concat = mat1[:] + mat2[:]
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        else:
            for row in range(len(mat1)):
                new_row = []
                for col in range(len(mat1[0])):
                    new_row.append(mat1[row][col])
                for col in range(len(mat2[0])):
                    new_row.append(mat2[row][col])
                concat.append(new_row)
    return concat
