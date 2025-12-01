#!/usr/bin/env python3
"""
Task 1: Trim Me Down
Complete the source code to extract specific columns from a matrix
"""
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = [row[2:4] for row in matrix]
print("The middle columns of the matrix are: {}".format(the_middle))
