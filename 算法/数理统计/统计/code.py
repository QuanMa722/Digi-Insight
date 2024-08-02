# -*- coding: utf-8 -*-

from scipy import stats
import numpy as np

# 给定的列表数据
data = [23, 45, 56, 78, 22, 19, 45, 67, 89, 12]

# 均值
mean = np.mean(data)
print(f"均值: {mean}")

# 方差
variance = np.var(data)
print(f"方差: {variance}")

# 标准差
std_dev = np.std(data)
print(f"标准差: {std_dev}")

# 最大值
max_value = np.max(data)
print(f"最大值: {max_value}")

# 最小值
min_value = np.min(data)
print(f"最小值: {min_value}")

# 极差
range_value = np.ptp(data)
print(f"极差: {range_value}")

# 中位数
median = np.median(data)
print(f"中位数: {median}")

# p分位数 (例如取第75百分位数)
p_quantile = np.percentile(data, 75)
print(f"第75百分位数 (p分位数): {p_quantile}")

# 众数
mode = stats.mode(data)
print(f"众数: {mode.mode}，出现次数: {mode.count}")

# 变异系数
coeff_variation = np.std(data) / np.mean(data) * 100
print(f"变异系数: {coeff_variation:.2f}%")
