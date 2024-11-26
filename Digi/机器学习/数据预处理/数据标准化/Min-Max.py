# -*- coding: utf-8 -*-

from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 示例数据
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12],
              [13, 14, 15]])

# 最小-最大标准化（Min-Max Scaling）
scaler = MinMaxScaler()
X_minmax = scaler.fit_transform(X)
print("最小-最大标准化:\n", X_minmax)

