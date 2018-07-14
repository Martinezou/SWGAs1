# -*- coding: utf-8 -*-
from numpy import *
import numpy as np


def calculate_modularity(a, b, n, c):
    """a-->process_code
       b-->machine_code
       n-->单元数
       c-->machine_unit机器所属单元"""
    workpiece = [[] for i in range(n)]
    for i in range(len(a)):
        for j in range(n):
            if a[i] == j:
                workpiece[j].append(b[i])
    A = zeros((67, 67))
    for i in range(len(workpiece)):
        for j in range(len(workpiece[i])):
            if j+1 < len(workpiece[i]):
                A[workpiece[i][j]][workpiece[i][j+1]] += 1
    A = mat(A)
    b1 = zeros((67, 20))
    num = 0
    d = 0
    for i in range(n):
        d += c[i]
        while num < d:
            b1[num][i] = 1
            num += 1
    b2 = b1.T
    m = (A.sum(axis=1)).sum(axis=0)[0, 0] / 2
    k = A.sum(axis=1)
    B = A - (k * k.T) * (float(1 / m)) * 0.5
    d = b2 * B * b1
    modularity1 = 1/m*np.trace(d)*0.5
    return modularity1
list1 = [14, 7, 11, 16, 5, 15, 0, 5, 14, 1, 2, 12, 11, 14, 18, 16, 2, 2, 17, 18, 9, 14, 4, 13, 7, 0, 8, 3, 8, 4, 10,
         13, 12, 17, 17, 1, 8, 1, 18, 8, 15, 7, 13, 1, 13, 9, 3, 11, 15, 3, 6, 10, 2, 8, 4, 5, 19, 6, 11, 15, 16, 19, 7,
         14, 19, 6, 5, 9, 3, 0, 19, 2, 9, 10, 12, 17, 11]
list2 = [0, 64, 50, 46, 27, 37, 64, 58, 27, 15, 37, 46, 11, 28, 45, 14, 44, 35, 4, 4, 23, 50, 11, 31, 20, 1, 0, 61, 24,
         33, 53, 42, 7, 33, 59, 66, 18, 61, 2, 63, 46, 24, 9, 54, 57, 11, 41, 42, 49, 3, 10, 26, 11, 3, 9, 54, 9, 32,
         41, 13, 30, 4, 32, 66, 34, 26, 13, 39, 13, 3, 6, 46, 49, 2, 3, 14, 54]
c = [4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 3]
print calculate_modularity(list1, list2, 20, c)

