#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/13 1:38 PM
# @Author  : HaoWang
# @Site    : 
# @File    : simulation_a.py
# @Software: PyCharm

import random as rdm

import numpy as np
from matplotlib import pyplot as plt
from numpy import cos, sin
from numpy import pi
from mpl_toolkits.mplot3d import Axes3D

# the random stream also input

def ipaestimator(theta, u):
    x = -  theta * np.log(1.0 - u)
    hder = (1.0 - 0.5 * x * (1.0 + 1.0 / (x + 1.0))
            / (theta * np.sqrt(x + 1.0)))
    return np.mean(hder)


def get_random_value(num=10, type='normal', mu=0, sigma=1, lamda=1, u=1):
    #     type = 'normal','uniform','gauss', 'exp'
    distribution = {'normal',
                    'gauss',
                    'uniform',
                    'exp'}
    if (type in distribution):
        result = np.zeros(num, dtype=np.float32)

        for i in range(num):
            if (type == 'normal' or type == 'gauss'):
                rdm_val = rdm.gauss(mu, sigma)
            elif (type == 'uniform'):
                rdm_val = rdm.uniform(0, u)
            else:
                rdm_val = rdm.expovariate(lamda)
            result[i] = rdm_val
        # print("Result: ",
        #       type,
        #       result,
        #       np.shape(result))
    else:
        print("Error: distribution type does not exist.")
        return -1
    return result


def r_func(theta1, theta2, n=3):
    r_theta = np.zeros((1, n), np.float32)

    r_theta[0, 0] = cos(theta1)
    r_theta[0, 1] = cos(theta2) * sin(theta1)
    r_theta[0, 2] = sin(theta2) * sin(theta1)

    return r_theta


def dr_func(theta1, theta2, n=3):
    dr_theta1 = np.zeros((1, n), np.float32)
    dr_theta2 = np.zeros((1, n), np.float32)

    dr_theta1[0, 0] = -sin(theta1)
    dr_theta1[0, 1] = cos(theta2) * cos(theta1)
    dr_theta1[0, 2] = sin(theta2) * cos(theta1)

    dr_theta2[0,0] = 0
    dr_theta2[0,1] = -sin(theta2) * sin(theta1)
    dr_theta2[0,2] = cos(theta2) * sin(theta1)

    return dr_theta1,dr_theta2

def EYi_Xi(mini_batch):
    xi = np.array([[2], [3], [1]])
    Xi = np.zeros((3, mini_batch), dtype=np.float32)
    Yi = np.zeros((3, mini_batch), dtype=np.float32)
    EYi = np.zeros((1, 3), dtype=np.float32)

    norm_V = get_random_value(mini_batch, type='normal', mu=0, sigma=1)
    exp_W = get_random_value(mini_batch, type='exp', lamda= 1/0.3)

    for i in range(len(exp_W)):
        # max W --> exp_W[]
        if (exp_W[i] <= 1):
            exp_W[i] = 1

    norm_I = np.zeros((3, mini_batch), dtype=np.float32)
    #  get random yitai, i=1,2,3 and calculate Xi

    for i in range(3):
        sum = 0
        cunter = 0
        norm_I[i, :] = get_random_value(mini_batch, type='normal', mu=0, sigma=(i + 1))
        Xi[i, :] = (0.6 * norm_V + np.sqrt(1 - 0.6 ** 2) * norm_I[i, :]) / exp_W
        Xi[i, :] = np.maximum(Xi[i, :], 0)
        for j in range(len(Xi[i, :])):
            if (Xi[i, j] >= xi[i, :]):
                sum += Xi[i, j]
                cunter += 1
        if cunter == 0:
            mean_X = 0
        else:
            mean_X = sum / cunter
        Yi[i, :] = get_random_value(mini_batch, type='uniform', u=mean_X)
        EYi[0, i] = (0 + mean_X) / 2
    # print('dim Xi:', Xi.ndim, Xi.shape, '\n', Xi)
    # print('dim Yi:', Yi.ndim, Yi.shape, '\n', Yi)
    return EYi


def J_theta(th1, th2, mini_batch=10):
    #  J_theta

    EYi = EYi_Xi(mini_batch)
    r_theta = r_func(theta1=th1, theta2=th2)
    j = np.dot(np.power(r_theta, 2),
                     EYi.T)
    return j

def gradient(th1,th2, mini_batch= 10):
    #  dJ_theta
    EYi_1 = EYi_Xi(mini_batch)
    EYi_2 = EYi_Xi(mini_batch)
    dj_1 = np.zeros((1, 1), dtype=np.float32)
    dj_2 = np.zeros((1, 1), dtype=np.float32)  # dr
    dr_1, dr_2 = dr_func(th1,th2)
    print('dr_1,2', dr_1, dr_2)
    r_theta = r_func(theta1=th1, theta2=th2)  #r(theta)
    print('r_theta', r_theta)

    for iter in range(3):
        dj_1 += (2* r_theta[0, iter])*EYi_1[0, iter]*dr_1[0,iter]
        dj_2 += (2 * r_theta[0, iter]) * EYi_2[0, iter] * dr_2[0, iter]
    print('dj',dj_1,dj_2)
    return dj_1,dj_2

def main():
    max_iterations = 2000
    mini_batch = 10  # samples per iteration
    th1 = np.zeros((1,max_iterations),dtype=np.float32)
    th2 = np.zeros((1,max_iterations),dtype=np.float32)  # initial value of theta

    ep = 0.01  # stepsize per iteration
    th1[0,0] = 0.01
    th2[0,0] = 0.01
    for i in range(max_iterations-1):
        dj_1, dj_2 = gradient(th1[0,i],th2[0,i],mini_batch=mini_batch)
        th1[0,i+1] = th1[0,i] + dj_1* ep
        th2[0,i+1] = th2[0,i] + dj_2* ep

    print(th1[0,max_iterations-1],th2[0,max_iterations-1])

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # th1, th2 = np.meshgrid(th1, th2)
    #
    # ax.scatter(th1, th2, j)
    #
    # # 添加坐标轴(顺序是Z, Y, X)
    # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    # plt.show()
    plt.plot(np.arange(max_iterations),
             th1.reshape(max_iterations,),
             color = 'r',
             label = 'theta1')
    plt.plot(np.arange(max_iterations),
             th2.reshape(max_iterations, ),
             color='b',
             label='theta2')

    plt.title('Theta1 and Theta2', fontsize=18, fontweight='bold')
    plt.xlabel(r'$iterations$', fontsize=11)
    plt.ylabel(r'$Theta$', fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=11, loc=1)
    plt.grid()
    plt.show()

def main1():
    max_iterations = 100
    mini_batch = 10  # samples per iteration
    th1 = np.linspace(0., pi/2,max_iterations).reshape((1,max_iterations))
    th2 = np.linspace(0., pi/2,max_iterations).reshape((1,max_iterations))  # initial value of theta
    j_theta = np.zeros((max_iterations,max_iterations))
    for i in range(max_iterations):
        for j in range(max_iterations):
            j_theta[i,j] = J_theta(th1[0,i],th2[0,j],mini_batch=mini_batch)
    print('J.dim',j_theta.shape,j_theta)
    print(np.max(j_theta))
if __name__ == '__main__':
    main()
