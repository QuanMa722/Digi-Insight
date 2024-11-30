# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform, loguniform
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np

# 读取数据
for cluster in range(0, 3):

    df = pd.read_excel(f'cluster_{cluster}.xlsx')
    X = df[['f1', 'f2', 'f4']]
    y = df['target']

    # 数据归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

    # 创建 CatBoost 模型
    model = CatBoostRegressor(silent=True)

    # 参数搜索空间
    param_dist = {
        'iterations': [50, 100, 200],  # Discrete values
        'learning_rate': loguniform(1e-3, 1e-1),  # Log-uniform distribution for learning rate
        'depth': [6, 8, 10, 12, 14],  # Discrete values for depth
        'l2_leaf_reg': uniform(1, 10),  # Uniform distribution from 1 to 11
        'subsample': uniform(0.7, 0.3),  # Uniform distribution from 0.7 to 1.0
        'border_count': [32, 64, 128, 256]  # Discrete values for border count
    }

    # 设置随机搜索
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=100,  # 迭代次数
        scoring='neg_mean_squared_error',  # 对于回归任务使用负均方误差
        cv=5,  # 交叉验证折数
        random_state=42,
        n_jobs=-1
    )

    # 执行随机搜索
    random_search.fit(X_train, y_train)

    # 使用最佳参数的模型进行预测
    best_model = random_search.best_estimator_
    y_pred_scaled = best_model.predict(X_test)

    # 逆归一化预测结果和真实值
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # 计算评估指标
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r_mse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    score = best_model.score(X_test, y_test)

    print(f'Cluster_{cluster}')
    print(f"R^2: {score:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R_MSE: {r_mse:.4f}")
    print()
