# -*- coding: utf-8 -*-

from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# 设置字体和负号显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def generate_data(samples=300, centers=4, std_dev=0.60, random_state=0):
    """Generate synthetic dataset using make_blobs."""
    return make_blobs(n_samples=samples, centers=centers, cluster_std=std_dev, random_state=random_state)


def calculate_silhouette_scores(X, K_max=15):
    """Calculate silhouette scores for a range of cluster counts."""
    scores = []
    for k in range(2, K_max):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        scores.append(score)
    return scores


def plot_silhouette_scores(scores, K_max):
    """Plot silhouette scores."""
    plt.subplot(1, 2, 1)
    plt.plot(range(2, K_max), scores, marker='o')
    plt.title('轮廓系数')
    plt.xlabel('聚类数')
    plt.ylabel('平均轮廓分')
    plt.grid()


def plot_clusters_with_boundaries(X, kmeans, best_k_num):
    """Plot clusters with decision boundaries."""
    h = 0.02  # step size of the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.subplot(1, 2, 2)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')

    labels = kmeans.labels_
    unique_labels = np.unique(labels)
    for i in unique_labels:
        plt.scatter(X[labels == i, 0], X[labels == i, 1], marker='o', label=f'Cluster {i + 1}', edgecolor='k')

    # Plot centroids
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, label='Centroids')

    plt.title('K-Means 聚类')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.legend()


def main():
    # 生成数据
    X, _ = generate_data()

    # 计算轮廓系数
    K_max = 15
    silhouette_scores = calculate_silhouette_scores(X, K_max)

    # 找到最佳聚类数
    best_k_num = np.argmax(silhouette_scores) + 2  # +2 because we started from k=2
    print(f"The best k-num: {best_k_num}")

    plt.figure(figsize=(10, 5))

    # 绘制轮廓系数
    plot_silhouette_scores(silhouette_scores, K_max)

    # 使用最佳聚类数进行K-Means聚类
    kmeans = KMeans(n_clusters=best_k_num, random_state=0)
    kmeans.fit(X)

    # 绘制聚类和决策边界
    plot_clusters_with_boundaries(X, kmeans, best_k_num)

    plt.show()


if __name__ == "__main__":
    main()
