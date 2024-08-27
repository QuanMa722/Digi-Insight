# -*- coding: utf-8 -*-

from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取数据
df1 = pd.read_excel("data.xlsx", sheet_name='sheet1')
df2 = pd.read_excel("data.xlsx", sheet_name='sheet2')

# 提取特征
X1 = df1[['f1', 'f2', 'f4']]
X2 = df2[['f1', 'f2', 'f4']]

# 创建一个新的图形
fig = plt.figure()

# 添加3D坐标轴
ax = fig.add_subplot(111, projection='3d')

# 绘制第一个数据集
ax.scatter(X1['f1'], X1['f2'], X1['f4'], c='r', label='Data from Sheet1', alpha=0.6)

# 绘制第二个数据集
ax.scatter(X2['f1'], X2['f2'], X2['f4'], c='k', label='Data from Sheet2', alpha=0.6)

# 设置坐标轴标签
ax.set_xlabel('F1')
ax.set_ylabel('F2')
ax.set_zlabel('F4')

# 添加图例
ax.legend()
# 显示图形
plt.show()


# 计算长方体内的点数
def count_points_within_box(box, data):
    f1_min, f1_max, f2_min, f2_max, f4_min, f4_max = box
    return np.sum((data['f1'] >= f1_min) & (data['f1'] <= f1_max) &
                  (data['f2'] >= f2_min) & (data['f2'] <= f2_max) &
                  (data['f4'] >= f4_min) & (data['f4'] <= f4_max))


# 目标函数：最小化黑点数，同时最大化红点数
def objective_function(box):
    red_points_in_box = count_points_within_box(box, X1)
    black_points_in_box = count_points_within_box(box, X2)
    return black_points_in_box - red_points_in_box  # 最小化黑点数


# 边界约束：确保边界值合理
bounds = [(X1['f1'].min(), X1['f1'].max()),
          (X1['f1'].min(), X1['f1'].max()),
          (X1['f2'].min(), X1['f2'].max()),
          (X1['f2'].min(), X1['f2'].max()),
          (X1['f4'].min(), X1['f4'].max()),
          (X1['f4'].min(), X1['f4'].max())]

# 初始猜测：选择合理的初始值
initial_guess = [X1['f1'].min(), 42.5,
                 X1['f2'].min(), 17,
                 2.8, 3.2]

# 第三问优化
result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')

# 提取每个变量的范围
f1_min, f1_max = result.x[0], result.x[1]
f2_min, f2_max = result.x[2], result.x[3]
f4_min, f4_max = result.x[4], result.x[5]

# 打印优化结果
print(f"f1范围: [{f1_min:.2f}, {f1_max:.2f}]")
print(f"f2范围: [{f2_min:.2f}, {f2_max:.2f}]")
print(f"f4范围: [{f4_min:.2f}, {f4_max:.2f}]")

# 计算优化结果下的红点数和黑点数
optimized_box = [f1_min, f1_max, f2_min, f2_max, f4_min, f4_max]
red_points_in_box = count_points_within_box(optimized_box, X1)
black_points_in_box = count_points_within_box(optimized_box, X2)

# 打印红点数和黑点数
print(f"优化结果下的红点数: {red_points_in_box}")
print(f"优化结果下的黑点数: {black_points_in_box}")
print(f"红点占比: {round(red_points_in_box / (red_points_in_box + black_points_in_box) * 100)}%")


# 在3D图中绘制长方体的函数
def draw_box(ax, box, color='b', alpha=0.3):
    f1_min, f1_max, f2_min, f2_max, f4_min, f4_max = box

    # 定义长方体的8个顶点
    vertices = np.array([[f1_min, f2_min, f4_min],
                         [f1_max, f2_min, f4_min],
                         [f1_max, f2_max, f4_min],
                         [f1_min, f2_max, f4_min],
                         [f1_min, f2_min, f4_max],
                         [f1_max, f2_min, f4_max],
                         [f1_max, f2_max, f4_max],
                         [f1_min, f2_max, f4_max]])

    # 定义长方体的12条边
    edges = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # 绘制长方体的边
    for edge in edges:
        ax.plot3D(*zip(*vertices[edge]), color=color, alpha=alpha)


# 筛选出在优化后的长方体内的红点和黑点
def filter_points_within_box(box, data):
    f1_min, f1_max, f2_min, f2_max, f4_min, f4_max = box
    mask = (data['f1'] >= f1_min) & (data['f1'] <= f1_max) & \
           (data['f2'] >= f2_min) & (data['f2'] <= f2_max) & \
           (data['f4'] >= f4_min) & (data['f4'] <= f4_max)
    return data[mask]


# 筛选在优化长方体内的红点和黑点
filtered_red_points = filter_points_within_box(optimized_box, X1)
filtered_black_points = filter_points_within_box(optimized_box, X2)

# 创建一个新的图形
fig = plt.figure()

# 添加3D坐标轴
ax = fig.add_subplot(111, projection='3d')

# 绘制在优化长方体内的红点
ax.scatter(filtered_red_points['f1'], filtered_red_points['f2'], filtered_red_points['f4'], c='r',
           label='Filtered Red Points', alpha=0.6)

# 绘制在优化长方体内的黑点
ax.scatter(filtered_black_points['f1'], filtered_black_points['f2'], filtered_black_points['f4'], c='k',
           label='Filtered Black Points', alpha=0.6)

# 绘制优化后的长方体
draw_box(ax, optimized_box, color='k', alpha=1)

# 设置坐标轴标签
ax.set_xlabel('F1')
ax.set_ylabel('F2')
ax.set_zlabel('F4')

# 添加图例
ax.legend()

# 显示图形
plt.show()
