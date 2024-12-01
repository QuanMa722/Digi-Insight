# -*- coding: utf-8 -*-

from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取数据
def load_data(file_path):
    df1 = pd.read_excel(file_path, sheet_name='Sheet1')
    df2 = pd.read_excel(file_path, sheet_name='Sheet2')
    return df1[['f1', 'f2', 'f4']], df2[['f1', 'f2', 'f4']]

# 计算长方体内的点数
def count_points_within_box(box, data):
    f1_min, f1_max, f2_min, f2_max, f4_min, f4_max = box
    return np.sum(
        (data['f1'] >= f1_min) & (data['f1'] <= f1_max) &
        (data['f2'] >= f2_min) & (data['f2'] <= f2_max) &
        (data['f4'] >= f4_min) & (data['f4'] <= f4_max)
    )

# 目标函数：最小化黑点数，同时最大化红点数
def objective_function(box, X1, X2):
    red_points_in_box = count_points_within_box(box, X1)
    black_points_in_box = count_points_within_box(box, X2)
    return black_points_in_box - red_points_in_box

# 边界约束：确保边界值合理
def get_bounds(X1):
    return [
        (X1['f1'].min(), X1['f1'].max()),
        (X1['f1'].min(), X1['f1'].max()),
        (X1['f2'].min(), X1['f2'].max()),
        (X1['f2'].min(), X1['f2'].max()),
        (X1['f4'].min(), X1['f4'].max()),
        (X1['f4'].min(), X1['f4'].max())
    ]

# 优化过程：最小化黑点数，同时最大化红点数
def optimize_box(X1, X2, initial_guess):
    bounds = get_bounds(X1)
    result = minimize(objective_function, initial_guess, args=(X1, X2), bounds=bounds, method='L-BFGS-B')
    return result.x

# 绘制3D图形的长方体
def draw_box(ax, box, color='b', alpha=0.3):
    f1_min, f1_max, f2_min, f2_max, f4_min, f4_max = box
    vertices = np.array([
        [f1_min, f2_min, f4_min], [f1_max, f2_min, f4_min], [f1_max, f2_max, f4_min], [f1_min, f2_max, f4_min],
        [f1_min, f2_min, f4_max], [f1_max, f2_min, f4_max], [f1_max, f2_max, f4_max], [f1_min, f2_max, f4_max]
    ])
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    for edge in edges:
        ax.plot3D(*zip(*vertices[edge]), color=color, alpha=alpha)

# 筛选在长方体内的点
def filter_points_within_box(box, data):
    f1_min, f1_max, f2_min, f2_max, f4_min, f4_max = box
    mask = (
        (data['f1'] >= f1_min) & (data['f1'] <= f1_max) &
        (data['f2'] >= f2_min) & (data['f2'] <= f2_max) &
        (data['f4'] >= f4_min) & (data['f4'] <= f4_max)
    )
    return data[mask]

# 主函数
def main():
    # 读取数据
    X1, X2 = load_data("data3.xlsx")

    # 初始猜测值
    initial_guess = [
        X1['f1'].min(), 42.5, X1['f2'].min() + 4, 17, 2.8, 3.1
    ]

    # 优化过程
    optimized_box = optimize_box(X1, X2, initial_guess)

    # 打印优化结果
    f1_min, f1_max, f2_min, f2_max, f4_min, f4_max = optimized_box
    print(f"f1范围: [{f1_min:.2f}, {f1_max:.2f}]")
    print(f"f2范围: [{f2_min:.2f}, {f2_max:.2f}]")
    print(f"f4范围: [{f4_min:.2f}, {f4_max:.2f}]")

    # 计算优化结果下的红点数和黑点数
    red_points_in_box = count_points_within_box(optimized_box, X1)
    black_points_in_box = count_points_within_box(optimized_box, X2)
    print(f"优化结果下的红点数: {red_points_in_box}")
    print(f"优化结果下的黑点数: {black_points_in_box}")
    print(f"红点数占比: {round(red_points_in_box / (red_points_in_box + black_points_in_box) * 100)}%")

    # 筛选在优化长方体内的红点和黑点
    filtered_red_points = filter_points_within_box(optimized_box, X1)
    filtered_black_points = filter_points_within_box(optimized_box, X2)

    # 创建一个新的图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制优化后的红点和黑点
    ax.scatter(filtered_red_points['f1'], filtered_red_points['f2'], filtered_red_points['f4'], c='r',
               label='Filtered Red Points', alpha=0.6)
    ax.scatter(filtered_black_points['f1'], filtered_black_points['f2'], filtered_black_points['f4'], c='k',
               label='Filtered Black Points', alpha=0.6)

    # 绘制优化后的长方体
    draw_box(ax, optimized_box, color='k', alpha=1)

    # 设置坐标轴标签
    ax.set_xlabel('F1')
    ax.set_ylabel('F2')
    ax.set_zlabel('F4')
    ax.set_title('Suboptimal solution optimization results')

    # 添加图例
    # ax.legend()

    # 显示图形
    plt.show()

if __name__ == "__main__":
    main()
