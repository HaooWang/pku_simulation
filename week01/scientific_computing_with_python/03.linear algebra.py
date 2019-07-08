#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/5 6:55 PM
# @Author  : HaoWang
# @Site    : 
# @File    : 03.linear algebra.py
# @Software: PyCharm

import numpy as np
import numpy.linalg as la

# 01. dot--向量点积
# To be used for column-row or row-column multiplications 行列相乘相加
# in vector-vector, matrix- vector,
# or matrix-matrix operations.
# Be careful about shapes and sizes.

A = np.arange(8).reshape(2, 4)
x = np.array([[2, 1]])  # row -vector

dot = np.dot(x, A) # x row-vector dot A 2*4 matrix  1*2 dot 2*4 = 1*4 1行4列向量
print(A, dot)

# 02. 向量内积 Inner product of two vectors.
# 对应位置相乘相加
a, b = np.array ([[1,3,5,8],[2,3,4,5]]), np.array ([[-2,4,-2,6],[4,5,6-7,8]])
inner = np.inner(a,b)
print(np.shape(a))
print(inner)

# 03. The linalg submodule of NumPy contains special linear algebra operations. First, import
# the submodule, e.g. by giving it a nickname:

# Cholesky decomposition of a symmetric positive-definite matrix (like variance-covariance
# matrix of multidimensional normal distribution).
# Cholesky分解对称正定矩阵（如方差 - 协方差 多维正态分布矩阵）。
Sigma = np.array ([[1.0 , -0.6] ,[ -0.6 ,1.0]])
cho = la.cholesky(Sigma)
print(cho)

# 04. eig() 特征值与特征向量
# la.eig(A): computes eigenvalues and eigenvectors of a square matrix A, or la.eigvals(A)
# for only the eigenvalues.
c = np.append(a, b, axis=0)
print(c)
eigvals = la.eigvals(c)   # 特征值
eigvec = la.eig(c)  # 特征根
print(eigvec)
print(eigvals)

# 05. norm矩阵范数
# norm(...)
# la.norm(x,ord=<n>): n-th norm of vector or matrix. For vectors, n 2 Z or np.inf,
# for matrices n = 1; 2 or np.inf.
norm1 = la.norm(c, 1, axis=0)  # 1范数为模之和
norm2 = la.norm(c, 2, axis=0)  # 2范数
print(norm1)
print(norm2)

# 06.
# cond(...)
# la.cond(A): computes condition number of matrix A.
cond = la.cond(c)
print(cond)
print(np.mean(cond))   # condition number == mean value of matrix C

# 07.
# det(...)
# la.det(A): computes the determinant-- 行列式 of a square matrix A. 方阵
det = la.det(c)
print(det)
print(np.floor(det))  # 向下取整

# 08.
# inv(...)
# la.inv(A): computes the inverse of a square matrix A. 方阵
# Be careful about the shapes an sizes.
trans = c.T   # 方阵（矩阵）转置
inv = la.inv(c)  # 方阵求逆矩阵
print(trans)
print(inv)

# 09.
# solve(...)
# la.solve(A,b): solves Ax = b for square matrix A.

# 10.
# lstsq
# la.lstsq(X,y): returns the least-squares solution to a linear equation X = y. The
# output is (1) solution ; (2) the SSR (sum of squared residuals); (3) rank of X; (4) the
# singular values of X.
