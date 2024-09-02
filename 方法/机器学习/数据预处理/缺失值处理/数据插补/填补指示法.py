# -*- coding: utf-8 -*-
"""
“填补指示法”是一种处理缺失数据的方法，通常涉及为每个数据点生成一个指示变量（也称为缺失指示符）来标记缺失值，
然后利用这些指示变量来改善模型的预测能力。
"""

from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# 示例数据：包含缺失值的 DataFrame
data = {
    'Feature1': [1, 2, np.nan, 4, 5],
    'Feature2': [10, np.nan, 30, 40, np.nan]
}
df = pd.DataFrame(data)

# 生成缺失指示变量
df['Feature1_missing'] = df['Feature1'].isna().astype(int)
df['Feature2_missing'] = df['Feature2'].isna().astype(int)

# 使用均值填补缺失值
imputer = SimpleImputer(strategy='mean')
df[['Feature1', 'Feature2']] = imputer.fit_transform(df[['Feature1', 'Feature2']])

# 打印填补后的 DataFrame
print(df)
