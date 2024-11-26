# -*- coding: utf-8 -*-

from sklearn.preprocessing import PowerTransformer
import numpy as np

# 生成示例数据
X = np.array([[1, 2, 4],
              [2, 4, 8],
              [3, 6, 12],
              [4, 8, 16]])

# 定义 Box-Cox 转换器
boxcox_transformer = PowerTransformer(method='box-cox', standardize=True)

# 应用 Box-Cox 转换
X_boxcox_transformed = boxcox_transformer.fit_transform(X)

# 输出结果
print("原始数据:")
print(X)
print("\nBox-Cox 转换后的数据:")
print(X_boxcox_transformed)
