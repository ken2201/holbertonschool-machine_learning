#!/usr/bin/env python3
"""
Task 5: Across The Plane
Write a function that adds two matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise

    Args:
        mat1: first 2D matrix (list of lists of ints/floats)
        mat2: second 2D matrix (list of lists of ints/floats)

    Returns:
        a new matrix containing the element-wise sum
        None if mat1 and mat2 are not the same shape
    """
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
            for i in range(len(mat1))]
