# -*- coding: utf-8 -*-

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

# 设置字体以支持中文
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 示例数据
data = [10, 12, 14, 15, 16, 18, 20, 22, 24, 25, 30, 32, 40, 100]
data = np.array(data).reshape(-1, 1)  # 转换为二维数组

# 使用DBSCAN进行异常值检测
dbscan = DBSCAN(eps=5, min_samples=2)  # eps: 半径，min_samples: 领域内最小样本数
y_pred = dbscan.fit_predict(data)

# -1 表示噪声点（异常值），其他值表示簇标签
is_outlier = y_pred == -1

# 绘制散点图
x = range(len(data))
plt.scatter(x, data, label='Data Points')
plt.xlabel('索引', fontsize=12)
plt.ylabel('值', fontsize=12)
plt.title('DBSCAN异常值散点图', fontsize=14)

# 高亮显示异常值
outlier_indices = np.where(is_outlier)[0]
plt.scatter(outlier_indices, data[is_outlier], color='red', label='异常点')

plt.legend(fontsize=10)
plt.show()
