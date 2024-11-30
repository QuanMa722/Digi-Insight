# -*- coding: utf-8 -*-

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
from boruta import BorutaPy
import xgboost as xgb
import pandas as pd
import numpy as np
import shap

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
df = pd.read_excel("data2.xlsx")

# 分离特征和目标变量
X = df.drop(columns='target')
y = df['target']

# 获取变量名
feature_names = X.columns

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
y_pred = best_xgb_model.predict(X_test)

# 计算模型的R^2分数
score = best_xgb_model.score(X_test, y_test)
print('R^2: ', round(score, 4))

# 将标准化后的数据转换为DataFrame，并恢复特征名称
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# 计算SHAP值
explainer = shap.TreeExplainer(best_xgb_model)
shap_values = explainer.shap_values(X_scaled_df)  # 传入特征矩阵X，计算SHAP值

# 可视化解释
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
shap.summary_plot(shap_values, X_scaled_df)
shap.summary_plot(shap_values, X, plot_type='bar', color='#f6416c')


# # # 定义随机森林回归器
# rf_reg = RandomForestRegressor()
#
# # 使用Boruta进行特征选择
# boruta = BorutaPy(estimator=rf_reg, n_estimators='auto', random_state=42)
# boruta.fit(X_train, y_train)
#
# # 获取被选择的特征的布尔数组
# selected_features = boruta.support_
#
# # 获取被选择的特征的名称
# selected_feature_names = X.columns[selected_features]
# print(f'Boruta: {list(selected_feature_names)}')

# 绘制学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score (R^2)")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='r2')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Plot learning curve for the best XGBoost model
title = "Learning Curves (XGBoost)"
cv = 5
plot_learning_curve(random_search.best_estimator_, title, X_train, y_train, cv=cv, n_jobs=-1)
plt.show()
