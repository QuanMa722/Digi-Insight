# -*- coding: utf-8 -*-

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd
import numpy as np
import shap

# 设置字体和负号显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_excel("data.xlsx", sheet_name='Sheet2')

# 分离特征和目标变量
X = df.drop(columns='收盘价')
y = df['收盘价']

# 初始化MinMaxScaler对特征进行归一化
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

# 定义参数分布
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.1),
    'num_leaves': randint(20, 100),
    'min_child_samples': randint(1, 20),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5)
}

# 创建LightGBM回归器对象
lgb_reg = lgb.LGBMRegressor()
# 创建RandomizedSearchCV对象
random_search = RandomizedSearchCV(
    estimator=lgb_reg,  # 需要优化的模型，这里是 LGBMRegressor 实例
    param_distributions=param_dist,  # 超参数分布的字典，用于指定要搜索的超参数范围
    n_iter=100,  # 随机搜索的迭代次数，即从参数分布中随机选择多少组参数组合进行评估
    cv=5,  # 交叉验证的折数，即将数据划分为 5 份进行交叉验证
    scoring='r2',  # 评估模型性能的指标，这里使用 R² 评分（决定系数），用于回归任务
    verbose=1,  # 控制输出的详细程度，1 表示输出一些基本的信息
    random_state=42,  # 随机种子，用于保证每次运行时的结果一致
    n_jobs=-1  # 并行运行的作业数，-1 表示使用所有可用的 CPU 核心
)
# 执行随机搜索
random_search.fit(X_train, y_train)

# 使用最优参数的模型进行预测和评估
best_xgb_model = random_search.best_estimator_
y_pred_scaled = best_xgb_model.predict(X_test)

# 逆归一化预测结果和真实值
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# 计算MAE和RMSE
mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))

# 计算模型的R^2分数
score = best_xgb_model.score(X_test, y_test)

print('-' * 10)
print("R^2: ", round(score, 4))
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print('-' * 10)

explainer = shap.TreeExplainer(best_xgb_model)
shap_values = explainer.shap_values(X)  # 传入特征矩阵X，计算SHAP值

# 可视化解释
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
shap.summary_plot(shap_values, X)
shap.summary_plot(shap_values, X, plot_type="bar", color='#f30070')
