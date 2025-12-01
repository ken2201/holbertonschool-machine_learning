#!/usr/bin/env python3
"""
Task 8: Ridin' Bareback
Write a function that performs matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication

    Args:
        mat1: first 2D matrix (list of lists of ints/floats)
        mat2: second 2D matrix (list of lists of ints/floats)

    Returns:
        a new matrix containing the result of the multiplication
        None if the two matrices cannot be multiplied
    """
    # Check if matrices are valid and can be multiplied
    if not mat1 or not mat2 or not mat1[0] or not mat2[0]:
        return None
    # Check if matrices can be multiplied (columns of mat1 == rows of mat2)
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize result matrix with zeros
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Perform matrix multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
