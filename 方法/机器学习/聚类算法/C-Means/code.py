# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


class CMeans:
    def __init__(self, n_clusters=2, max_iter=100, m=2, error=1e-6):
        self.n_clusters = n_clusters  # 类别数
        self.max_iter = max_iter  # 最大迭代次数
        self.m = m  # 模糊度指数
        self.error = error  # 迭代停止误差阈值
        self.centers = None  # 聚类中心
        self.u = None  # 隶属度矩阵

    def fit(self, X):
        # 初始化隶属度矩阵
        n_samples = X.shape[0]
        self.u = np.random.rand(n_samples, self.n_clusters)
        self.u = self.u / np.sum(self.u, axis=1, keepdims=True)

        iteration = 0
        while iteration < self.max_iter:
            # 计算聚类中心
            self.centers = np.dot(self.u.T, X) / np.sum(self.u, axis=0, keepdims=True).T

            # 计算新的隶属度
            u_old = self.u.copy()
            dist = np.linalg.norm(X[:, None, :] - self.centers, axis=2) ** 2
            self.u = 1 / np.sum((dist[:, :, None] / dist[:, None, :]) ** (1 / (self.m - 1)), axis=2)

            # 检查误差是否足够小
            if np.linalg.norm(self.u - u_old) < self.error:
                break

            iteration += 1

    def predict(self, X):
        # 预测数据点的类别
        dist = np.linalg.norm(X[:, None, :] - self.centers, axis=2) ** 2
        u_pred = 1 / np.sum((dist[:, :, None] / dist[:, None, :]) ** (1 / (self.m - 1)), axis=2)
        return np.argmax(u_pred, axis=1)


# 示例用法
if __name__ == "__main__":
    # 生成一些随机数据
    np.random.seed(0)
    n_samples = 300
    centers = [[1, 1], [-1, -1], [1, -1]]
    X = np.random.randn(n_samples, 2) + centers[0]
    X = np.r_[X, np.random.randn(n_samples, 2) + centers[1]]
    X = np.r_[X, np.random.randn(n_samples, 2) + centers[2]]

    # 使用C-means聚类
    cmeans = CMeans(n_clusters=3)
    cmeans.fit(X)

    # 预测类别
    y_pred = cmeans.predict(X)

    # 可视化结果
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=20, cmap='viridis')
    plt.scatter(cmeans.centers[:, 0], cmeans.centers[:, 1], marker='^', s=100, c='r', label='cluster centers')
    plt.title('C-Means 聚类')
    plt.legend()
    plt.show()
