# -*- coding: utf-8 -*-

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# 读取数据
df = pd.read_excel("data2.xlsx")

# 提取特征
X = df[['f1', 'f2', 'f4']]

# 创建 KMeans 模型并进行拟合
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
df['cluster'] = kmeans.labels_

# 创建一行两个子图
fig = plt.figure(figsize=(12, 6))

# 第一个子图：绘制原始散点图
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X['f1'], X['f2'], X['f4'])
ax1.set_xlabel('F1')
ax1.set_ylabel('F2')
ax1.set_zlabel('F4')
ax1.set_title('Original Data')

# 第二个子图：绘制聚类后的散点图
ax2 = fig.add_subplot(122, projection='3d')
scatter = ax2.scatter(df['f1'], df['f2'], df['f4'], c=df['cluster'], cmap='viridis', marker='o')
ax2.set_xlabel('F1')
ax2.set_ylabel('F2')
ax2.set_zlabel('F4')
ax2.set_title('KMeans Clustering')

# 显示图形
plt.show()
