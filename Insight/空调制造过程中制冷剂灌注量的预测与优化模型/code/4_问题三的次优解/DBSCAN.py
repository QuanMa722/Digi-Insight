# -*- coding: utf-8 -*-

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取数据
df1 = pd.read_excel("data3.xlsx", sheet_name='Sheet1')
df2 = pd.read_excel("data3.xlsx", sheet_name='Sheet2')

# 提取特征
X1 = df1[['f1', 'f2', 'f4']]
X2 = df2[['f1', 'f2', 'f4']]

# 创建一个新的图形
fig = plt.figure()

# 添加3D坐标轴
ax = fig.add_subplot(111, projection='3d')

# 绘制第一个数据集的点
scatter1 = ax.scatter(X1['f1'], X1['f2'], X1['f4'], c='#ff2e63', label='Points in the range', alpha=0.6)

# 绘制第二个数据集的点
scatter2 = ax.scatter(X2['f1'], X2['f2'], X2['f4'], c='#393e46', label='Points outside the range', alpha=0.6)

# 设置坐标轴标签
ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.set_zlabel('f4')

# 添加图例
plt.legend()

# 显示图形
plt.show()

# 标准化特征
scaler = StandardScaler()

X1_scaled = scaler.fit_transform(X1)
X2_scaled = scaler.fit_transform(X2)

# 应用 DBSCAN 聚类
dbscan1 = DBSCAN(eps=0.5, min_samples=10)
labels1 = dbscan1.fit_predict(X1_scaled)

dbscan2 = DBSCAN(eps=0.5, min_samples=10)
labels2 = dbscan2.fit_predict(X2_scaled)

# 计算每个簇的点数
unique_labels1, counts1 = np.unique(labels1, return_counts=True)
unique_labels2, counts2 = np.unique(labels2, return_counts=True)

# 找到点数最多的簇标签
max_cluster_label1 = unique_labels1[np.argmax(counts1)]
max_cluster_label2 = unique_labels2[np.argmax(counts2)]

# 过滤出最大簇的数据
filtered_data1 = X1[labels1 == max_cluster_label1]
filtered_data2 = X2[labels2 == max_cluster_label2]

# 创建一个新的图形
fig = plt.figure()

# 添加3D坐标轴
ax = fig.add_subplot(111, projection='3d')

# 绘制第一个数据集的最大簇
ax.scatter(filtered_data1['f1'], filtered_data1['f2'], filtered_data1['f4'], c='#ff2e63', label='Max Density Cluster Scope1', alpha=0.6)

# 绘制第二个数据集的最大簇
ax.scatter(filtered_data2['f1'], filtered_data2['f2'], filtered_data2['f4'], c='#393e46', label='Max Density Cluster Scope2', alpha=0.6)

# 设置坐标轴标签
ax.set_xlabel('F1')
ax.set_ylabel('F2')
ax.set_zlabel('F4')

# 添加图例
plt.legend()

# 显示图形
plt.show()

