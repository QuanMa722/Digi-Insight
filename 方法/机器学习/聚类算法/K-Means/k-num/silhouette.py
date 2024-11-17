# -*- coding: utf-8 -*-

from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# 设置字体和负号显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

silhouette_scores = []
K_max = 15

for k in range(2, K_max):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)

best_k_num = list(range(2, K_max))[silhouette_scores.index(max(silhouette_scores))]
print(f"The best_k_num: {best_k_num}")

plt.plot(range(2, K_max), silhouette_scores, marker='o')
plt.title('轮廓系数')
plt.xlabel('聚类数')
plt.ylabel('平均轮廓得分')
plt.grid()
plt.show()
