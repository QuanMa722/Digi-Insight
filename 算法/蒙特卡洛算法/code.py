# -*- coding: utf-8 -*-

import numpy as np


def objective_function(x):
    return x[0] ** 2 + x[1] ** 2


def constraint(x):
    return x[0] + x[1] - 1


# 蒙特卡罗方法求解非线性规划问题
def monte_carlo_nonlinear_programming(obj_func, constraint_func, num_samples=1000000):
    min_value = np.inf
    min_point = None

    for _ in range(num_samples):
        x = np.random.rand(2) * 2  # 生成随机点，范围在 [0, 2] 之间
        if constraint_func(x) >= 0:  # 检查约束条件
            obj_value = obj_func(x)
            if obj_value < min_value:
                min_value = obj_value
                min_point = x

    return min_point, min_value


for num in range(1, 11):
    # 使用蒙特卡罗方法求解
    solution, min_obj_value = monte_carlo_nonlinear_programming(objective_function, constraint)
    print(f"目标函数解{num}:", solution)
    print(f"目标函数值{num}:", min_obj_value)
