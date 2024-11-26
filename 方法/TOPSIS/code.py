# -*- coding: utf-8 -*-

"""
A  89  3
B  60  1
C  74  2
D  99  4
"""

import numpy as np

# 给定数据矩阵
# + -
X = np.array([
    [89, 3],
    [60, 1],
    [74, 2],
    [99, 4]
])

# 正向化
X_process = np.array([
    [89, 1],
    [60, 3],
    [74, 2],
    [99, 0]
])

# 找出每列的最小值和最大值
min_vals = np.min(X_process, axis=0, keepdims=True)
max_vals = np.max(X_process, axis=0, keepdims=True)

# 进行0-1标准化
X_normalized = (X_process - min_vals) / (max_vals - min_vals)

# 打印标准化后的结果
# print("标准化后的数据矩阵:")
# print(X_normalized)

# 进行线性加权
for num in range(4):
    print(["A", "B", "C", "D"][num], 0.6 * X_normalized[num][0] + 0.4 * X_normalized[num][1])
