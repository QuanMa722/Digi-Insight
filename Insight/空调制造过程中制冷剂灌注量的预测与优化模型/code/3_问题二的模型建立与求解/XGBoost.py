# -*- coding: utf-8 -*-

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
for cluster in range(0, 3):
    df = pd.read_excel(f'cluster_{cluster}.xlsx')

    # 分离特征和目标变量
    X = df.drop(columns='target')
    y = df['target']

    # 初始化MinMaxScaler对特征进行归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

    # 定义参数分布
    param_dist = {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.1),
        'gamma': uniform(0, 0.5),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5)
    }

    # 创建XGBoost回归器对象
    xgb_reg = xgb.XGBRegressor()

    # 创建RandomizedSearchCV对象
    random_search = RandomizedSearchCV(
        estimator=xgb_reg,
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        scoring='r2',
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # 执行随机搜索
    random_search.fit(X_train, y_train)

    # 使用最优参数的模型进行预测和评估
    best_xgb_model = random_search.best_estimator_
    y_pred_scaled = best_xgb_model.predict(X_test)

    # 逆归一化预测结果和真实值
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # 计算MAE和R_MSE
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r_mse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))

    # 计算模型的R^2分数
    score = best_xgb_model.score(X_test, y_test)

    print(f'Cluster_{cluster}')
    print(f"R^2: {score:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R_MSE: {r_mse:.4f}")
    print()



