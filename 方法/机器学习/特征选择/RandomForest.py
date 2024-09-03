# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)
y_df = pd.Series(y, name='Target')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=42)

# 训练随机森林分类器
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 获取特征重要性
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# 绘制特征重要性图
plt.figure(figsize=(14, 7))

# 特征重要性条形图
plt.subplot(1, 2, 1)
sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')

# 随机森林模型的预测准确率
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 显示预测结果的混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()
