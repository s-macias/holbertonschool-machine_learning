#!/usr/bin/env python3
""" module to create a function to calculate the determinant of a matrix """


def determinant(matrix):
    """
    Function to calculate the determinant of a matrix
    Arguments:
    matrix is a list of lists whose determinant should be calculated
    The list [[]] represents a 0x0 matrix
    Returns:
    the determinant of matrix
    """
    rows = len(matrix)
    cols = len(matrix[0])
    det = 0

    if type(matrix) is not list or rows == 0 or matrix == []:
        raise TypeError("matrix must be a list of lists")
    if rows == 0:
        raise TypeError("matrix must be a list of lists")
    if rows == 1 and cols == 0:
        return 1
    for row in matrix:
        if type(row) is not list or len(row) == 0:
            raise TypeError("matrix must be a list of lists") if len(row) != rows:
            raise ValueError("matrix must be a square matrix")

    if rows == 1:
        return matrix[0][0]
    det = 0
    if rows == 2:
        return det2x2(matrix)

    submatrices = list(range(len(matrix)))
    for sub in submatrices:
        sub_matrix = []
        for i in range(rows):
            sub_matrix.append(matrix[sub])
        sub_matrix = matrix[1:]
        rows_sub_m = len(sub_matrix)

        for row in range(rows_sub_m):
            matrix[row] = matrix[row][0:sub] + matrix[row][sub+1:]
        sign = (-1) ** (sub)
        sub_det = determinant(sub_matrix)
        det += sign * matrix[0][i] * sub_det
    return det


def det2x2(matrix):
    """
    Calculates matrix of shape (2,2)
    Arguments:
    matrix is a list of lists
    Returns:
    determinant of the matrix
    """
    det = (matrix[0][0] * matrix[1][1]) - (matrix[1][0] * matrix[0][1])
    return det
