#!/usr/bin/env python3
"""
Task 14: Saddle Up
Write a function that performs matrix multiplication
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication using numpy

    Args:
        mat1: first numpy.ndarray
        mat2: second numpy.ndarray

    Returns:
        a new numpy.ndarray containing the result of the multiplication
    """
    return np.matmul(mat1, mat2)
