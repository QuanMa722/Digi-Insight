# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_excel("data1.xlsx")

# 绘制1行2列的子图
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 创建一个1行2列的子图布局

for idx, i in enumerate(['f1', 'f2']):
    data = df[i]

    x = np.arange(len(data))
    y = data

    # 判断哪些数据为负值，分别设置不同颜色
    colors = np.where(y < 0, '#e84545', '#8caacf')  # 负值设置为红色，其他为深蓝色

    # 绘制散点图
    axs[idx].scatter(x, y, alpha=1, color=colors, s=10)  # 设置点的大小为10
    axs[idx].set_title(f'{i} 散点图')

    # 设置x轴和y轴标签
    axs[idx].set_xlabel('index')
    axs[idx].set_ylabel('value')

plt.tight_layout()  # 自动调整子图布局
plt.show()
