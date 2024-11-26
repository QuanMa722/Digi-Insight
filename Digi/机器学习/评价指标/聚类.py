# -*- coding: utf-8 -*-

"""
轮廓系数（Silhouette Score）：衡量样本相似度，值范围为[-1, 1]，值越大表示聚类效果越好。
Calinski-Harabasz 指数：衡量簇内紧凑度和簇间分离度，值越大表示聚类效果越好。
Davies-Bouldin 指数：衡量簇的紧凑度与簇间分离度，值越小表示聚类效果越好。
"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 生成示例数据
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 进行 KMeans 聚类
kmeans = KMeans(n_clusters=4, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# 计算评价指标
silhouette_avg = silhouette_score(X, y_kmeans)
calinski_harabasz_avg = calinski_harabasz_score(X, y_kmeans)
davies_bouldin_avg = davies_bouldin_score(X, y_kmeans)

width = 20
print('-' * 30)
print(f"{'Silhouette Score:':<{width}}{silhouette_avg:.4f}")
print(f"{'Calinski-Harabasz:':<{width}}{calinski_harabasz_avg:.4f}")
print(f"{'Davies-Bouldin:':<{width}}{davies_bouldin_avg:.4f}")
print('-' * 30)
