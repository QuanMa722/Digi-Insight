import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_moons

# 创建一个合成数据集
X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)

# 创建一个支持向量机模型，使用RBF核和正则化参数
C = 1.0  # 正则化参数
model = svm.SVC(kernel='rbf', C=C)
model.fit(X, y)

# 创建一个网格来绘制决策边界
xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 500), np.linspace(-1, 1.5, 500))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制结果
plt.figure(figsize=(10, 6))

# 绘制决策边界和间隔
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred', alpha=0.3)

# 绘制训练点
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.spring, edgecolors='k')

# 绘制支持向量
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], facecolors='none', edgecolors='k', s=100, linewidths=1.5)

plt.title(f"Support Vector Machine with RBF Kernel and Regularization (C={C})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()