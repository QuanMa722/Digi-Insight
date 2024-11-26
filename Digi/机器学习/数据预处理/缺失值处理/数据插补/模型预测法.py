# -*- coding: utf-8 -*-
"""
模型预测法填补数据是一种使用机器学习模型来估计和填补数据集中缺失值的技术。
这种方法可以根据数据中的其他特征来预测缺失值，从而比简单的插值或均值填补方法提供更准确的填补。下
面是如何使用模型预测法填补数据的详细解释：
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# 示例数据：包含缺失值的 DataFrame
data = {
    'X': [1, 2, 3, 4, 5, 6],
    'y': [10, np.nan, 30, np.nan, 50, 60]  # y 列中有缺失值
}

df = pd.DataFrame(data)

# 将数据分为两部分：有缺失值的和无缺失值的
df_missing = df[df['y'].isna()]
df_not_missing = df.dropna()

# 特征和目标
X_not_missing = df_not_missing[['X']]
y_not_missing = df_not_missing['y']

# 创建和训练模型
model = LinearRegression()
model.fit(X_not_missing, y_not_missing)

# 对缺失值进行预测
X_missing = df_missing[['X']]
predicted_y = model.predict(X_missing)

# 填补缺失值
df.loc[df['y'].isna(), 'y'] = predicted_y

print(df)
