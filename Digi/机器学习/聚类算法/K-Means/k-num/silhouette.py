# -*- coding: utf-8 -*-

from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# 设置字体和负号显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 模拟数据集 centers=3
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# 计算不同聚类数下的轮廓系数
silhouette_scores = []
K_max = 15

for k in range(2, K_max):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)

# 获取最佳聚类数
best_k_num = np.argmax(silhouette_scores) + 2
print(f"最佳聚类数: {best_k_num}")

# 绘制轮廓系数图
plt.plot(range(2, K_max), silhouette_scores, marker='o')
plt.title('轮廓系数')
plt.xlabel('聚类数')
plt.ylabel('平均轮廓得分')
plt.grid(True)
plt.show()


