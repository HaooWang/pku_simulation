#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/5 5:36 PM
# @Author  : HaoWang
# @Site    : 
# @File    : 02.Numpy array routines.py
# @Software: PyCharm

import numpy as np

#  array manipulation routines
# transpose, concatenate, delete, insert, append,
# flip, etc;

x = np.array([[1.0, 2, 3],
              [-4, 5, 6],
              [-7.0, 8, 9]])

y = np.zeros((3, 3), dtype=float)

# 01. A.transpose() or its alias A.T.
xt = x.T  # Transpose --  be careful with memory management!
xt = x.transpose()

# 02. np.delete(<array>,<index>,<axis>) deletes the item at index of axis, 0,1,2.
x_del = np.delete(x, 1, 0)  # delete the index '1' axis--row（0）

# 03. np.append(<array>,<object>,axis=<n>) appends object at the end of axis of array.
# Be careful about rank and sizes.
x_append_row = np.append(x, y, axis=0)  # 追加行
x_append_col = np.append(x, y, axis=1)  # 追加列

# 04. sort
# np.sort(x,0): sorts the elements of x along axis 0; thus for a rank 2 array, it produces
# a matrix with sorted (low-to-high) columns; idem np.sort(x,1), np.sort(x), etc.
A = np.random.rand(3, 4)
A_sort = np.sort(A, axis=0)  # 按行进行小到大排序
print(A_sort)

# 05. amax（）
# np.amax(x,0): maximum of the elements of x along axis 0; thus for a rank 2 array,
# it produces a row with the column-maxima.
# Alias is np.max. idem np.amax(x,1),
# np.amax(x), etc.
amax = np.amax(A_sort, axis=0)  # 每一列的最大值
amin = np.amin(A_sort, axis=0)  # 每一列的最小值
print("amax:", amax)
print("amin:", amin)

#  06. mean()
# np.mean(x,0): the average of the elements of x along axis 0; thus for a rank 2 array,
# it produces a row with the column-averages; idem np.mean(x,1), np.mean(x), etc.
mean = np.mean(A_sort, axis=0)
print("mean:", mean)

# 07. histogram
# np.histogram(x): computes the histogram of the elements of x. Default is 10 equalwidth
# bins defined by the range of the data. You can change this by the option
# bins=<integer> or bins=<array> for specifying the bin-edges.
rdm = np.random.rand (1000)  # 1000 U(0,1) random numbers
h = np.histogram(rdm, bins=15)
print(h[0])

# ---- Information Printing ---
print("---- Information Printing ---")
print("x    : \n", x)
print("y    : \n", y)
print("xt   : \n", xt)

print("x_del    : \n", x_del)

print("x_append_row : \n", x_append_row)
print("x_append_col : \n", x_append_col)
# x_append_row :
#  [[ 1.  2.  3.]
#  [-4.  5.  6.]
#  [-7.  8.  9.]
#  [ 0.  0.  0.]
#  [ 0.  0.  0.]
#  [ 0.  0.  0.]]
# x_append_col :
#  [[ 1.  2.  3.  0.  0.  0.]
#  [-4.  5.  6.  0.  0.  0.]
#  [-7.  8.  9.  0.  0.  0.]]
