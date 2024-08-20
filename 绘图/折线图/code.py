# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# 示例数据
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# 创建折线图
plt.plot(x, y, marker='o', linestyle='--', color='#b83b5e')

# 去掉上边框和右边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 显示图形
plt.show()
