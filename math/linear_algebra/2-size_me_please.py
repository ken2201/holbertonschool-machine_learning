#!/usr/bin/env python3
"""
Task 2: Size Me Please
Write a function that calculates the shape of a matrix
"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix

    Args:
        matrix: a matrix (list of lists)

    Returns:
        list of integers representing the shape of the matrix
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) > 0:
            matrix = matrix[0]
        else:
            break
    return shape
