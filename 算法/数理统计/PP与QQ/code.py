# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 生成正态分布的随机样本数据
np.random.seed(0)
data = np.random.normal(loc=0, scale=1, size=100)

# 计算排序后的累积分布函数值
sorted_data = np.sort(data)
cum_probs = np.linspace(0, 1, len(sorted_data))

# 计算理论正态分布的累积分布函数值
norm_probs = stats.norm.cdf(sorted_data, loc=0, scale=1)

# 计算样本数据和理论正态分布的分位数
norm_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)), loc=0, scale=1)

# 绘制PP图和QQ图放在一起
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# PP图
ax1.plot(norm_probs, cum_probs, marker='o', linestyle='none')
ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 对角线
ax1.set_title('概率-概率图（PP图）')
ax1.set_xlabel('理论累积概率')
ax1.set_ylabel('样本累积概率')
ax1.grid(True)

# QQ图
ax2.scatter(norm_quantiles, sorted_data, color='#1f77b4')
ax2.plot([-3, 3], [-3, 3], color='gray', linestyle='--')  # 对角线
ax2.set_title('定量-定量图 （QQ图）')
ax2.set_xlabel('理论定量（标准正态分布）')
ax2.set_ylabel('样本定量')
ax2.grid(True)

plt.tight_layout()
plt.show()
