#!/usr/bin/env python3
"""
Task 12: Bracing The Elements
Write a function that performs element-wise addition, subtraction,
multiplication, and division
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division

    Args:
        mat1: first numpy.ndarray
        mat2: second numpy.ndarray

    Returns:
        tuple containing the element-wise sum, difference, product, and quotient
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
