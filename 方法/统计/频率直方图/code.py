# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 生成随机数据作为示例
np.random.seed(0)
data = np.random.normal(loc=50, scale=10, size=10000)  # 正态分布数据

# 绘制频率直方图
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=30, kde=False, stat='density', linewidth=0.5)

# 添加正态分布曲线
mean = np.mean(data)
std_dev = np.std(data)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mean, std_dev)
plt.plot(x, p, 'gray', linewidth=2, linestyle='--')

# 设置标题和标签
plt.title('频率直方图与正态分布曲线')
plt.xlabel('数据值')
plt.ylabel('密度')

# 显示图例
plt.legend(['正态分布曲线', '频率直方图'])

# 显示图形
plt.show()
