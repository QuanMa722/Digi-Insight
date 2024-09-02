# -*- coding: utf-8 -*-

from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# 设置字体和负号显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

silhouette_scores = []
K_max = 15
for k in range(2, K_max):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)

best_k_num = list(range(2, K_max))[silhouette_scores.index(max(silhouette_scores))]
print("-" * 20)
print(f"The best k-num: {best_k_num}")
print("-" * 20)

plt.figure(figsize=(10, 5))

# Plot Silhouette scores
plt.subplot(1, 2, 1)
plt.plot(range(2, K_max), silhouette_scores, marker='o')
plt.title('轮廓系数')
plt.xlabel('聚类数')
plt.ylabel('平均轮廓分')
plt.grid()

# Plot clusters with decision boundaries
plt.subplot(1, 2, 2)

h = 0.02  # step size of the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

kmeans = KMeans(n_clusters=best_k_num)
kmeans.fit(X)
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')

labels = kmeans.predict(X)
for i in range(4):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], cmap='viridis', marker='o', label=f'Cluster {i+1}', edgecolor='k')

# Plot centroids
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, label='Centroids')

plt.title('带决策边界的 K-Means 聚类法')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.legend()
plt.show()
