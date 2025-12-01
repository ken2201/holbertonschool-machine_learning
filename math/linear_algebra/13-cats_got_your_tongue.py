#!/usr/bin/env python3
"""
Task 13: Cat's Got Your Tongue
Write a function that concatenates two matrices along a specific axis
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two numpy.ndarrays along a specific axis

    Args:
        mat1: first numpy.ndarray
        mat2: second numpy.ndarray
        axis: axis along which to concatenate (default 0)

    Returns:
        a new numpy.ndarray containing the concatenation
    """
    return np.concatenate((mat1, mat2), axis=axis)
