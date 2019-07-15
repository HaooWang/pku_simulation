#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/10 5:45 PM
# @Author  : HaoWang
# @Site    : 
# @File    : SA_debug.py
# @Software: PyCharm


import random as rdm
import numpy as np
from matplotlib import pyplot as plt


# the random stream also input

def ipaestimator(theta, u):
    x = -  theta * np.log(1.0 - u)
    hder = (1.0 - 0.5 * x * (1.0 + 1.0 / (x + 1.0))
            / (theta * np.sqrt(x + 1.0)))
    return np.mean(hder)


def get_random_value(seed, num=10, type='normal', mu=0, sigma=1):
    #     type = 'normal','uniform','gauss', 'exp', 'log'
    distribution = {'normal', 'uniform', 'exp', 'log'}
    if (type in distribution):
        result = np.array((1, num), dtype=np.float32)
        for i in range(num):
            rdm_val = rdm.gauss(mu, sigma)
            np.append(result, rdm_val)
        print("Result: ", result)
    else:
        print("Error: distribution type does not exist.")


# get random value - shape = (1,m)
def get_random_value(theta, m):
    seed = 745795
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
def L_theta(c_list, m):
    sum = 0
    for i in range(m):
        sum += c_list[i]

    l_theta = sum / m
    return l_theta


# update theta i+1
def update_theta(G_theta, theta, ep, i):
    theta = theta - ep * G_theta / (i + 1)

    return theta


# Gradient theta i
def G_theta(L_theta, alpha=37):
    dire = -1
    g_theta = -(L_theta - alpha) * 1

    return g_theta


# ################################
# m = number of samples estimator per iteration
# n = number of iterations
# ep = epsilon fixed step size
# #################################

def main():
    altha = 37
    th0 = 1.5  # initial value of theta
    n = 2000  # iterations
    m = 100  # samples per iteration
    ep = 0.01  # stepsize per iteration
    # n-dim theta vector initialization
    theta = np.zeros(n + 1, dtype=np.float32)
    theta[0] = th0
    kesi = np.zeros((n, m), dtype=np.float32)
    lthetas = np.zeros(n, dtype=np.float32)
    GThetas = np.zeros(n, dtype=np.float32)
    for i in range(n):
        # #########
        # Step 1. （using the initial theta0 ）to generate one-group kesi（size = （1，m））；
        # Step 2. Calculate Xn， Cn and L_theta by using those kesi[] in step01,
        # and L_theta will be saved in lthetas
        # Step 3. do gradient estimator
        # Step 4. update a new theta named theta【i+1】
        # Step 5. if i< max_iterations, do next iteration Step1-5.
        # ###########

        kesi[i, :] = get_random_value(theta[i], m)
        Xn = X_list(kesi[i, :], m)
        Cn = C_list(Xn, kesi[i, :], m)
        lthetas[i] = L_theta(Cn, m)
        ######### update theta 【i+1】 ############
        GThetas[i] = G_theta(lthetas[i])
        theta[i + 1] = update_theta(GThetas[i], theta[i], ep, i)
        ######### update theta 【i+1】 ###########
    print("Theta Star: ", theta[n])
    plt.plot(lthetas,
             color='r',
             label="L(theta)",
             linewidth=1)
    plt.plot(theta,
             color='b',
             label="Theta",
             linewidth=1)
    altha = altha * np.ones(n)
    plt.plot(altha,
             color='k',
             label='altha',
             linewidth=1)
    # plt.autoscale(tight= True)
    # theta_test = np.arange(0.05,0.25,0.001)
    # len = 200
    # altha = altha * np.ones(len)
    # print(theta_test,len)
    # lthetas_test = np.zeros(len)
    # xn_test = np.zeros(len)
    # cn_test = np.zeros(len)
    # kesi_test = np.zeros((len,m))
    # for i in range(len):
    #     kesi_test[i,:] = get_random_value(theta_test[i],m)
    #     xn_test = X_list(kesi_test[i,:],m)
    #     cn_test = C_list(xn_test, kesi_test[i,:],m)
    #     lthetas_test[i] = L_theta(cn_test,m)
    #
    # print(lthetas_test)
    # plt.plot(theta_test,
    #         lthetas_test,
    #          color = 'r',
    #          label = 'ltheta')
    # plt.plot(theta_test,
    #          altha,
    #          color = 'b',
    #          label = 'altha')
    plt.title('Graph of long-run cost and iterations', fontsize=18, fontweight='bold')
    plt.xlabel(r'$iterations$', fontsize=11)
    plt.ylabel(r'$L(theta)$', fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=11, loc=1)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
