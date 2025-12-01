#!/usr/bin/env python3
"""
Task 9: Let The Butcher Slice It
Complete the source code to slice various matrices using numpy
"""
import numpy as np
matrix = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
                   [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]])
mat1 = matrix[:3, :]
mat2 = matrix[1:3, :]
mat3 = matrix[:, 2:4]
print("The first three rows of the matrix are: {}".format(mat1))
print("The middle two rows of the matrix are: {}".format(mat2))
print("The middle columns of the matrix are: {}".format(mat3))
