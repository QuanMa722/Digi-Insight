# -*- coding: utf-8 -*-
# 经验分布函数提供了数据的累积分布信息，可以直观地理解数据的分布形状和集中程度。
# 通过观察经验分布函数的形状，可以初步判断数据是否服从某种已知的分布模型，如正态分布、指数分布等。

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 示例数据
data = np.random.normal(size=1000)

# 绘制经验分布函数图
sns.ecdfplot(data, linewidth=2)
plt.title('经验分布函数')
plt.xlabel('数据')
plt.ylabel('累积概率')
plt.grid(True)
plt.show()
