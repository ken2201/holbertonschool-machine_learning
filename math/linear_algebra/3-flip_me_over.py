#!/usr/bin/env python3
"""
Task 3: Flip Me Over
Write a function that returns the transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix

    Args:
        matrix: a 2D matrix (list of lists)

    Returns:
        a new matrix that is the transpose of the input matrix
    """
    if not matrix or not matrix[0]:
        return []
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
