# -*- coding: utf-8 -*-

from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# 设置环境变量 LOKY_MAX_CPU_COUNT 为 4
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# 读取数据
df = pd.read_excel("data2.xlsx")

# 提取特征
X = df[['f1', 'f2', 'f4']]

# 设置要测试的簇数范围
range_n_clusters = list(range(2, 8))  # 从2到7

silhouette_avg_list = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_

    # 计算轮廓系数
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_avg_list.append(silhouette_avg)
    print(f'Number of clusters = {n_clusters}, Average silhouette score = {round(silhouette_avg, 2)}')

# 绘制轮廓系数图
plt.figure(figsize=(8, 6))
plt.plot(range_n_clusters, silhouette_avg_list, marker='o', linestyle='--', c='#b83b5e')
plt.xlabel('Number of clusters')
plt.ylabel('Average silhouette score')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.grid()
plt.show()

# 根据最佳轮廓系数选择聚类数
best_n_clusters = range_n_clusters[np.argmax(silhouette_avg_list)]
print(f'Best number of clusters: {best_n_clusters}')

# 使用最佳簇数进行最终的K-means聚类
kmeans = KMeans(n_clusters=best_n_clusters)
kmeans.fit(X)
df['cluster'] = kmeans.labels_

# 创建1行3列的子图
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

# 绘制三个相同的图
for i, ax in enumerate(axes):
    scatter = ax.scatter(df['f1'], df['f2'], df['f4'], c=df['cluster'], cmap='viridis', marker='o')

    # 设置标签和标题
    ax.set_xlabel('F1')
    ax.set_ylabel('F2')
    ax.set_zlabel('F4')
    ax.set_title(f'Silhouette-K-Means')

# 显示图形
plt.tight_layout()
plt.show()

# 保存每个类别的数据到不同的Excel文件
for cluster_label in df['cluster'].unique():

    # 选择当前类别的数据
    cluster_data = df[df['cluster'] == cluster_label]

    # 创建文件名
    filename = f'cluster_{cluster_label}.xlsx'

    # 保存到Excel文件
    cluster_data.to_excel(filename, index=False)
    print(f'Saved data for cluster {cluster_label} to {filename}')
