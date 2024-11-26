# -*- coding: utf-8 -*-

from sklearn.preprocessing import StandardScaler
import numpy as np

# 示例数据
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12],
              [13, 14, 15]])

# Z-score 标准化（标准差标准化）
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
print("Z-score 标准化:\n", X_standardized)
