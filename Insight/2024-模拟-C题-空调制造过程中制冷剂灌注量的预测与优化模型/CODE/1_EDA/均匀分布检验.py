# -*- coding: utf-8 -*-

from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import pandas as pd

# 读取数据
data = pd.read_excel("data1.xlsx")

# 标准化数据到 [0, 1] 范围
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['f1', 'f2', 'f4', 'f5', 'f6', 'f7', 'f8', 'f10', 'f11', 'f12']])

# 显著性水平
alpha = 0.025

# 执行 Kolmogorov-Smirnov 检验
for i, col in enumerate(['f1', 'f2', 'f4', 'f5', 'f6', 'f7', 'f8', 'f10', 'f11', 'f12']):
    # 执行 Kolmogorov-Smirnov 检验
    ks_statistic, p_value = stats.kstest(data_scaled[:, i], 'uniform')

    # 判断检验结果
    if p_value >= alpha:
        print(f"变量 {col} 服从均匀分布 (p-value: {p_value:.4f})")
        print()
