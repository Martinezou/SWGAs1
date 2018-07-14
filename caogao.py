# -*- coding: utf-8 -*-
import numpy as np
from SWGAs import*


def calculate_max_makespan(a, b, c):
    """计算每个零部件的完工时间并挑选出最大完工时间
           a-->process_code工序编码
           b-->machine_code机器编码
           c-->workpiece_machine_time每个零部件在机器上的加工时间
           d-->机器之间的运输时间"""
    matrix1 = np.zeros((67, 20))    # 根据工序编码和机器编码生成每个零部件在每个机器上的加工时间
    for i in range(len(a)):
        try:
            matrix1[(b[i], a[i])] = c[(a[i], b[i])]
        except KeyError:
            pass
    matrix2 = np.zeros((67, 20))   # 根据工序编码和机器编码生成每个零部件在每个机器上的完工时间
    distance = 0
    for i in range(len(a)):
        try:
            max_column = matrix2.max(axis=0)  # 每列最大值
            max_row = matrix2.max(axis=1)  # 每行最大值
            for j in range(i-1, -1, -1):     # 求出当前工序和紧前工序加工机器之间的距离
                if a[i] == a[j]:
                    p = b[i]
                    q = b[j]
                    distance = machine_distance(p, q)
                    pass
            matrix2[(b[i], a[i])] = max(max_column[a[i]]+distance, max_row[b[i]]) + c[(a[i], b[i])]

        except KeyError:
            pass
    return matrix2
    # 生成每个零部件在每个机器上的完工时间





process_time_matrix = calculate_max_makespan(process_code, machine_code, workpiece_machine_time)
print process_time_matrix
#print workpiece_machine_time[(20,3)]
