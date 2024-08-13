# -*- coding: utf-8 -*-

"""
网格搜索（Grid Search）:通过指定一个超参数的取值范围，并对这些参数的所有可能组合进行穷举，从而找到模型的最佳参数配置。
"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes

# 生成示例数据
data = load_diabetes()
X, y = data.data, data.target

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义随机森林回归模型
model = RandomForestRegressor()

# 定义超参数的搜索空间
param_grid = {
    'n_estimators': [50, 100, 200],  # 树的数量
    'max_depth': [None, 10, 20, 30],  # 最大深度
    'min_samples_split': [2, 5, 10],  # 分裂节点的最小样本数
    'min_samples_leaf': [1, 2, 4],  # 叶子节点的最小样本数
    'bootstrap': [True, False]  # 是否使用自助采样
}

# 定义网格搜索
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,       # 使用 5 折交叉验证
    scoring='r2',  # 评估模型性能的指标
    verbose=1,  # 控制输出的详细程度
    n_jobs=-1,  # 使用所有可用的 CPU 核心
)

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
score = best_model.score(X_test, y_test)
print("R^2:", round(score, 4))
