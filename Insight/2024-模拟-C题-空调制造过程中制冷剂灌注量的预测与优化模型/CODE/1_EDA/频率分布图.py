# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel("data1.xlsx")

fig, axs = plt.subplots(2, 4, figsize=(18, 10))  # 创建一个2行4列的子图布局
for idx, i in enumerate(['f4', 'f5', 'f6', 'f10', 'f7', 'f8', 'f11', 'f12']):
    data = df[i]

    # 计算正态分布曲线参数
    mean = np.mean(data)
    std_dev = np.std(data)
    xmin, xmax = np.min(data), np.max(data)
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean, std_dev)

    # 绘制频率直方图和正态分布曲线
    row = idx // 4
    col = idx % 4
    axs[row, col].hist(data, bins=30, density=True, alpha=0.6, color='#3f72af')
    axs[row, col].set_title(f'{i} 频率直方图')

    if idx == 0 or idx == 1 or idx == 2 or idx == 3:
        axs[row, col].plot(x, p, 'gray', linewidth=2, linestyle='--')
        axs[row, col].legend(['正态分布曲线'], loc='upper left')
        axs[row, col].grid()

plt.tight_layout()  # 自动调整子图布局
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 创建一个1行2列的子图布局
for idx, i in enumerate(['f1', 'f2']):
    data = df[i]

    # 计算正态分布曲线参数
    mean = np.mean(data)
    std_dev = np.std(data)
    xmin, xmax = np.min(data), np.max(data)
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean, std_dev)

    # 绘制频率直方图和正态分布曲线
    col = idx  # 因为是一行两列，所以列数直接用 idx
    axs[col].plot(x, p, 'gray', linewidth=2, linestyle='--')
    axs[col].hist(data, bins=30, density=True, alpha=0.6, color='#3f72af')
    axs[col].set_title(f'{i} 频率直方图')
    axs[col].grid()

plt.tight_layout()  # 自动调整子图布局
plt.show()


