# -*- coding: utf-8 -*-

from sklearn.preprocessing import Normalizer
import numpy as np

# 示例数据
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12],
              [13, 14, 15]])

# 单位向量标准化（Unit Vector Scaling）
scaler = Normalizer()
X_normalized = scaler.fit_transform(X)
print("单位向量标准化:\n", X_normalized)

