# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import pandas as pd
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

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

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

