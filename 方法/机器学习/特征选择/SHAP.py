# -*- coding: utf-8 -*-

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint, uniform
import xgboost as xgb
import pandas as pd
import numpy as np
import shap

# 读取数据
df = pd.read_excel("data.xlsx")

# 分离特征和目标变量
X = df.drop(columns='target')
y = df['target']

# 标准化之前保存特征名称
feature_names = X.columns

# 初始化MinMaxScaler对特征进行归一化
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# 划分数据
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
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# 计算MAE和RMSE
mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))

# 计算模型的R^2分数
score = best_xgb_model.score(X_test, y_test)

print('-' * 12)
print("R^2: ", round(score, 4))
print('-' * 12)

# 将标准化后的数据转换为DataFrame，并恢复特征名称
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

explainer = shap.TreeExplainer(best_xgb_model)
shap_values = explainer.shap_values(X_scaled_df)  # 传入特征矩阵X，计算SHAP值

# 可视化解释
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
shap.summary_plot(shap_values, X_scaled_df)
shap.summary_plot(shap_values, X_scaled_df, plot_type="bar", color='#f30070')


