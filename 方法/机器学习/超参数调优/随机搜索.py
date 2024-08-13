# -*- coding: utf-8 -*-

"""
随机搜索（Random Search）是一种超参数优化方法，用于在给定的超参数空间中寻找最佳模型配置。
"""

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.datasets import load_diabetes
from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestRegressor

# 生成示例数据
data = load_diabetes()
X, y = data.data, data.target

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义随机森林回归模型
model = RandomForestRegressor()

# 定义超参数的搜索空间
param_dist = {
    'n_estimators': randint(50, 201),  # 树的数量从 50 到 200 之间的整数
    'max_depth': [None, 10, 20, 30, 40, 50],  # 最大深度的候选值
    'min_samples_split': randint(2, 21),  # 分裂节点的最小样本数从 2 到 20 之间的整数
    'min_samples_leaf': randint(1, 21),  # 叶子节点的最小样本数从 1 到 20 之间的整数
    'bootstrap': [True, False]  # 是否使用自助采样
}

# 定义随机搜索
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,  # 随机搜索的迭代次数
    cv=5,       # 使用 5 折交叉验证
    scoring='r2',  # 评估模型性能的指标
    verbose=1,  # 控制输出的详细程度
    random_state=42,  # 随机种子
    n_jobs=-1   # 使用所有可用的 CPU 核心
)

# 进行随机搜索
random_search.fit(X_train, y_train)

# 使用最佳模型进行预测
best_model = random_search.best_estimator_
score = best_model.score(X_test, y_test)
print("R^2:", round(score, 4))
