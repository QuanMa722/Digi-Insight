# -*- coding: utf-8 -*-

from sklearn.preprocessing import FunctionTransformer
import numpy as np

# 生成示例数据
X = np.array([[1, 10, 100],
              [2, 20, 200],
              [3, 30, 300],
              [4, 40, 400]])

# 定义对数转换函数
log_transformer = FunctionTransformer(np.log1p, validate=True)

# 应用对数转换
X_log_transformed = log_transformer.fit_transform(X)

# 输出结果
print("原始数据:")
print(X)
print("\n对数转换后的数据:")
print(X_log_transformed)
