# -*- coding: utf-8 -*-

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint, uniform
from sklearn.metrics import r2_score
from boruta import BorutaPy
import pandas as pd

# 读取数据
df = pd.read_excel("data.xlsx")

# 分离特征和目标变量
X = df.drop(columns='target')
y = df['target']

# 初始化MinMaxScaler对特征进行归一化
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

# 定义随机森林回归器
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# 使用Boruta进行特征选择
boruta = BorutaPy(estimator=rf_reg, n_estimators='auto', random_state=42)
boruta.fit(X_train, y_train)

# 获取被选择的特征
X_train_selected = boruta.transform(X_train)
X_test_selected = boruta.transform(X_test)

# 获取被选择的特征的布尔数组
selected_features = boruta.support_

# 获取被选择的特征的名称
selected_feature_names = X.columns[selected_features]
print("Selected features:", selected_feature_names)

# 定义参数分布
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': uniform(0.5, 0.5)
}

# 创建RandomizedSearchCV对象
random_search = RandomizedSearchCV(
    estimator=rf_reg,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='r2',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# 执行随机搜索
random_search.fit(X_train_selected, y_train)

# 使用最优参数的模型进行预测和评估
best_rf_model = random_search.best_estimator_
y_pred = best_rf_model.predict(X_test_selected)

# 计算模型的R^2分数
score = r2_score(y_test, y_pred)

print('-' * 12)
print("R^2:", round(score, 4))
print('-' * 12)
