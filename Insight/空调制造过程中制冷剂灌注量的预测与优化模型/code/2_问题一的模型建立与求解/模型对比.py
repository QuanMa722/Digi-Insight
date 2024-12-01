# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd

# 用于存储不同数据量下每个模型的 R^2 分数
xgb_score_list = []
dt_score_list = []
rf_score_list = []

# 循环从 500 到 6500 数据量，按步长 500 增加
for num in range(1, 15):
    # 读取数据
    df = pd.read_excel("data1.xlsx")

    # 按照样本数选择数据
    df = df.sample(n=num * 500, random_state=42)

    # 分离特征和目标变量
    X = df.drop(columns='target')
    y = df['target']

    # 初始化MinMaxScaler对特征进行归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()  # 目标变量归一化

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

    # 定义参数空间并进行随机搜索

    # XGBoost参数优化
    xgb_param_grid = {
        'n_estimators': randint(100, 1000),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 15),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    }
    xgb_search = RandomizedSearchCV(xgb.XGBRegressor(), xgb_param_grid, n_iter=50, cv=3, random_state=42, n_jobs=-1)
    xgb_search.fit(X_train, y_train)
    xgb_score = xgb_search.score(X_test, y_test)

    # 决策树参数优化
    dt_param_grid = {
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 10)
    }
    dt_search = RandomizedSearchCV(DecisionTreeRegressor(random_state=42), dt_param_grid, n_iter=50, cv=3, random_state=42, n_jobs=-1)
    dt_search.fit(X_train, y_train)
    dt_score = dt_search.score(X_test, y_test)

    # 随机森林参数优化
    rf_param_grid = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 10),
        'bootstrap': [True, False]
    }
    rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, n_iter=50, cv=3, random_state=42, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    rf_score = rf_search.score(X_test, y_test)

    # 打印每个模型的结果
    print(f'Datasize {num * 500}')
    print(f'XGBoost R^2 score: {xgb_score:.4f}')
    print(f'Decision Tree R^2 score: {dt_score:.4f}')
    print(f'Random Forest R^2 score: {rf_score:.4f}')
    print()

    # 存储每个模型的R^2得分
    xgb_score_list.append(xgb_score)
    dt_score_list.append(dt_score)
    rf_score_list.append(rf_score)


# 数据绘图
plt.figure()

# 数据点
data_sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]

# 绘制每个模型的曲线
plt.plot(data_sizes, xgb_score_list, marker='o', linestyle='--', color='#e84545', label='XGBoost')
plt.plot(data_sizes, rf_score_list, marker='^', linestyle='--', color='#f9ed69', label='Random Forest')
plt.plot(data_sizes, dt_score_list, marker='s', linestyle='--', color='#6a2c70', label='Decision Tree')

# 显示图形
plt.xlabel('Data Size')
plt.ylabel('R^2 score')
plt.title('R^2 vs. Data Size with Hyperparameter Tuning')
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.show()
