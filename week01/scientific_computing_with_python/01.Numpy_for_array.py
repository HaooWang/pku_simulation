#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/3 8:20 PM
# @Author  : HaoWang
# @Site    : 
# @File    : 01.Numpy_for_array.py
# @Software: PyCharm

import numpy as np

# from matplotlib import pyplot as plt

# creat np.float64 array
x = np.array([[1.0, 2, 3],
              [-4, 5, 6],
              [-7.0, 8, 9],
              [0.0, 9, 8],
              [9, 8, 7]])

# reshape np.array
x1 = np.reshape(x, (3, 5))
x2 = np.reshape(x, (1,15))
# string
str = np.array([[1.0, 2.0, 3.0],
                ['1', '2', '3'],
                ['test', 'char', 'array']])
# matrix
arr_ones = np.ones((3, 3), dtype=np.float64)

# ---- Information Printing ---
print("---- Information Printing ---")
print("str : \n", str)
print("x    : \n", x)
print("x.dtype: ", x.dtype)
print("x.shape: ", x.shape)
print("str.dtype: ", str.dtype)
print("arr_ones: \n", arr_ones)

print("x.ndim:", x.ndim)  # x.ndim dimension of x array : 2D
print("x.shape: ", x.shape)
print("x.sizw: ", x.size)
# reshape x array to 3*5
print("x1.shape:", x1.shape)
print("x1: \n", x1)
print(x2[:,:6])

# result:
#         x.ndim: 2
#         x.shape:  (5, 3)
#         x.sizw:  15
#         x1.shape: (3, 5)
#         x1:
#          [[ 1.  2.  3. -4.  5.]
#          [ 6. -7.  8.  9.  0.]
#          [ 9.  8.  9.  8.  7.]]

