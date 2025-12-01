#!/usr/bin/env python3
"""
Task 4: Line Up
Write a function that adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise

    Args:
        arr1: first array (list of ints/floats)
        arr2: second array (list of ints/floats)

    Returns:
        a new list containing the element-wise sum
        None if arr1 and arr2 are not the same shape
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
