# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from boruta import BorutaPy
import pandas as pd

# 读取数据
df = pd.read_excel("data.xlsx")

# 分离特征和目标变量
X = df.drop(columns='target')
y = df['target']

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
print(f'重要的特征：{list(selected_feature_names)}')
