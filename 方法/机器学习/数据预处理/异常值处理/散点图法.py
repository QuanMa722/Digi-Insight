# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# 设置字体以支持中文
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

data = [10, 12, 14, 15, 16, 18, 20, 22, 24, 25, 30, 32, 40, 100]

# 示例数据
x = range(len(data))
y = data

# 绘制散点图
plt.scatter(x, y, label='Data Points')
plt.xlabel('索引', fontsize=12)
plt.ylabel('值', fontsize=12)
plt.title('识别异常值的散点图', fontsize=14)
plt.show()

