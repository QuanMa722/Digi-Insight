# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from boruta import BorutaPy
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
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

# 定义随机森林回归器
rf_reg = RandomForestRegressor()

# 使用Boruta进行特征选择
boruta = BorutaPy(estimator=rf_reg, n_estimators='auto', random_state=42)
boruta.fit(X_train, y_train)

# 获取被选择的特征的布尔数组
selected_features = boruta.support_

# 获取被选择的特征的名称
selected_feature_names = X.columns[selected_features]
print(f'BORUTA: {list(selected_feature_names)}')

# 创建XGBoost回归器对象并设置超参数
xgb_reg = xgb.XGBRegressor()

# 训练模型
xgb_reg.fit(X_train, y_train)

# 将标准化后的数据转换为DataFrame，并恢复特征名称
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# 计算SHAP值
explainer = shap.TreeExplainer(xgb_reg)
shap_values = explainer.shap_values(X_scaled_df)  # 传入特征矩阵X，计算SHAP值

# 可视化解释
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
shap.summary_plot(shap_values, X_scaled_df)

# 计算每个特征的平均绝对SHAP值
shap_importance = np.abs(shap_values).mean(axis=0)

# 创建一个包含特征名称和对应重要性的DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': shap_importance
})

# 按照特征的重要性从高到低排序
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 打印特征重要性排名
print("\nSHAP Feature Importance Ranking:")
print(importance_df)