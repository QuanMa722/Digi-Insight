# -*- coding: utf-8 -*-

"""
Optuna 是一个高效的自动超参数优化库，支持多种优化算法和优化策略。

CMA-ES 是一种进化策略（Evolution Strategy），主要用于连续优化问题。它特别适用于高维、非线性、不规则的优化问题。CMA-ES 通过迭代的方式调整搜索策略，以找到最优解。它的核心思想是使用一个协方差矩阵来表示搜索空间中个体的分布，从而动态调整搜索方向和步长。
TPE 是一种基于贝叶斯优化的方法，主要用于处理离散和连续超参数的优化问题。TPE 是一种概率模型优化算法，它基于贝叶斯统计的思想，通过建立超参数空间的概率模型来指导搜索过程。

CMA-ES 更加适合于需要全局优化和高维连续空间的情况。
TPE 适合于具有复杂结构的超参数优化，能够有效地处理具有不同类型超参数的情况。
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import optuna

# 生成示例数据
data = load_diabetes()
X, y = data.data, data.target

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def objective(trial):
    # 定义超参数空间
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 10, 50, step=10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }

    # 定义模型
    model = RandomForestRegressor(**param, random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测并评估
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)


# 创建 Optuna 的 study 对象
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# 输出最佳超参数和最佳得分
print("R^2:", round(study.best_value, 4))

