#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/9 5:28 PM
# @Author  : HaoWang
# @Site    : 
# @File    : SA.py
# @Software: PyCharm

import numpy as np
import numpy.random as rnd
from matplotlib import pyplot as plt


# the random stream also input

def ipaestimator(theta, u):
    x = -  theta * np.log(1.0 - u)
    hder = (1.0 - 0.5 * x * (1.0 + 1.0 / (x + 1.0))
            / (theta * np.sqrt(x + 1.0)))
    return np.mean(hder)


# get random value - shape = (1,m)
def get_random_value(theta, m):
    seed = 45923
    rnd.seed(seed)
    u = rnd.rand(m)
    kesi = np.zeros(m, dtype=np.float32)
    kesi = -  1 / theta * np.log(1.0 - u)
    return kesi


# Xn function
def X_list(kesi, m, S=10, s=5):  # kesi is m-length list
    x = np.zeros(m)
    x[0] = S
    for i in range(m - 1):
        if x[i] >= (kesi[i] + s):
            x[i + 1] = x[i] - kesi[i]
        else:
            x[i + 1] = S
    return x  # row vector


# C() function
def C_list(X_n, kesi, m,
           s=5,
           h=2,
           p=8,
           K=10):
    cost_C = np.zeros(m)
    for i in range(m):
        if X_n[i] < kesi[i]:
            cost_C[i] = p * (kesi[i] - X_n[i]) + K
        elif (X_n[i] - s) >= kesi[i]:
            cost_C[i] = h * (X_n[i] - kesi[i])
        else:
            cost_C[i] = h * (X_n[i] - kesi[i]) + K

    return cost_C


# L(theta) function
def L_theta(c_list):
    dim = np.shape(c_list)
    sum = 0
    for i in range(dim):
        sum += c_list[i]

    l_theta = sum / dim
    return l_theta


# batch gradient descent
def bgd(samples, theta, altha, step_size, max_iter_count=100):
    # initialization
    sample_num, dim = samples.shape
    kesi = np.zeros(dim, dtype=np.float32)
    loss = 1
    iter_count = 0

    while loss > 0.01 and iter_count < max_iter_count:
        loss = 0
        error = np.zeros((dim), dtype=np.float32)

        Xn = X_list(kesi, dim)
        Cn = C_list(Xn, kesi, dim)
        Ln = L_theta(Cn)

        for i in range(sample_num):
            error = (1 / 2) * np.power((Ln[i] - altha), 2)
            loss += error  # update loss function
            for j in range(dim):
                # decreasing epsilon
                theta[j] += step_size * error[j]

        print("iter_count: ", iter_count, "the loss:", loss)
        iter_count += 1
    return theta


def stochasticapproximation(theta, altha, n, m, ):

    pass
# ################################
# m = number of samples estimator per iteration
# n = number of iterations
# ep = epsilon fixed step size
# #################################

def main():
    altha = 37
    th0 = 0.06   # initial value of theta
    n = 100  # iterations
    m = 1000  # samples per iteration
    ep = 0.1  # stepsize per iteration
    # n-dim theta vector initialization
    theta = np.zeros(n, dtype=np.float32)
    theta[0] = th0
    kesi = np.zeros((n,m), dtype=np.float32)
    for i in range(n):
        kesi[i,:] = get_random_value(theta[i], m)
        Xn = X_list(kesi[i, :], m)
        Cn = C_list(Xn, kesi[i, :], m)
        L_theta = L_theta(Cn)
        theta[i+1] = theta########
        # update theta 【i+1】
        #######


    plt.plot(theta)
    plt.show()


if __name__ == '__main__':
    main()
