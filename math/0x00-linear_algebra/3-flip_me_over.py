#!/usr/bin/env python3
""" Returns transpose of a 2D matrix """


def matrix_transpose(matrix):
    """ returns the transpose of a 2D matrix, matrix """
    matrix_t = []
    rows = len(matrix)
    columns = len(matrix[0])
    for col in range(columns):
        column = []
        for row in range(rows):
            column.append(matrix[row][col])
        matrix_t.append(column)
    return matrix_t
