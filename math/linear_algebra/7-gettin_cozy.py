#!/usr/bin/env python3
"""
Task 7: Gettin' Cozy
Write a function that concatenates two matrices along a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis

    Args:
        mat1: first 2D matrix (list of lists of ints/floats)
        mat2: second 2D matrix (list of lists of ints/floats)
        axis: axis along which to concatenate (0 or 1)

    Returns:
        a new matrix containing the concatenation
        None if the two matrices cannot be concatenated
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    return None
