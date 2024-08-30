# -*- coding: utf-8 -*-

"""
均方误差 (MSE)：衡量模型预测值与实际值之间的平均平方差。值越小表示模型越好。
平均绝对误差 (MAE)：预测值与实际值之间绝对差的平均值。
决定系数 (R²)：表示模型解释目标变量方差的比例。值范围从 0 到 1，值越接近 1 表示模型越好。

R²衡量模型解释了多少目标变量的方差。它的值范围从 0 到 1，越接近 1 表示模型拟合越好。
然而，R²可能会随着自变量的增加而增大，即使新增加的自变量对模型并没有实际意义。
故采用调整后的 R²。
"""

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# 生成示例数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 对测试集进行预测
y_pred = model.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = pow(mse, 0.5)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# 计算平均绝对误差 (MAE)

# 打印结果
print("均方误差 (MSE):", round(mse, 4))
print("均方根误差 (RMSE):", round(rmse, 4))
print("平均绝对误差 (MAE):", round(mae, 4))
print("决定系数 (R²):", round(r2, 4))

n = X_test.shape[0]  # 样本数量
p = X_test.shape[1]  # 自变量数量
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# 打印结果
print("调整后的决定系数 (Adjusted R²):", round(r2_adj, 4))
