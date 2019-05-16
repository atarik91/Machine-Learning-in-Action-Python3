# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 13:06:53 2018

@author: Administrator
"""

from numpy import *
from os import  *
import builtins

def loadData(filename):
    '''
    '''
    datamat = []; labelmat = []
    with builtins.open(filename) as fr:
        for line in fr.readlines():
            line_arr = line.strip().split()
            datamat.append([1.0, float(line_arr[0]), float(line_arr[1])])
            # jisuan x0, x1,x2. x0wei 1
            labelmat.append(int(line_arr[2]))
    return datamat, labelmat

def sigmoid(inp):
    return 1.0 / (1 + exp(-inp))


    '''
        批量梯度下降
    '''
def Grad_descent(datamat, labels):

    data = mat(datamat)
    label = mat(labels).transpose()
    
    m, n = shape(datamat)
    alpha = 0.001; max_iter = 500
    weights = ones((n, 1))   # 
    for k in range(max_iter):
        #将每个特征值都乘以一个特征系数,然后将所有的特征值相加
        z = dot(datamat, weights)
        #带入sigmoid函数,得到0~1之间的数(算是归一化?)
        y_pred = sigmoid(z)
        error = (label - y_pred)
        # grad(x) = (y - f(x)) * x'  更新回归系数
        weights = weights + alpha * data.transpose() * error
    return weights

def test():

    num = [(1,2.5), (1.5, 3.2), (1.3, 4.0), (2.2, 1.8)]
    y,z = builtins.max(num, key=lambda x:x[0])
    max
    print(y, z)

    