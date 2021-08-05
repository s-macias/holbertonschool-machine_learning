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
    
    if type(matrix) is not list or rows == 0:
        raise TypeError("matrix must be a list of lists")
    if rows == 1 and cols == 0:
        return 1
    for row in matrix:
        if type(row) is not list or :
            raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError("matrix must be a square matrix")

    if rows == 1:
        return matrix[0][0]

    if rows == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[1][0] * matrix[0][1])

    submatrices = list(range(len(matrix)))
    for sub in submatrices:
        sub_matrix = matrix
        sub_matrix = sub_matrix[1:]
        for i in range(len(sub_matrix)):
            sub_matrix[i] = sub_matrix[i][:sub] + sub_matrix[i][sub+1:] 
        sub_det = determinant(sub_matrix)
        det += -1 * matrix[0][i] * sub_det
    return det
