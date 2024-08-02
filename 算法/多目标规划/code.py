# -*- coding: utf-8 -*-
# 线性加权 (math.)

import numpy as np
import pulp as lp

# 定义 LP 问题设置
prob = lp.LpProblem("Multi-Objective Problem", lp.LpMaximize)

# 决定变量
x1 = lp.LpVariable('x1', lowBound=0, upBound=None, cat='Continuous')
x2 = lp.LpVariable('x2', lowBound=0, upBound=None, cat='Continuous')

# 目标函数
objective1 = 2 * x1 + 3 * x2
objective2 = x1 + 2 * x2

# 制约因素
prob += 0.5 * x1 + 0.25 * x2 <= 8
prob += 0.2 * x1 + 0.2 * x2 <= 4
prob += x1 + 5 * x2 <= 72
prob += x1 + x2 >= 10

# 分别解决每个目标的问题
prob.solve()

# 获取每个目标的最优值
max_obj_value = lp.value(objective1)
min_obj_value = lp.value(objective2)

print("Objective 1 - Maximize 2x1 + 3x2:")
print("Status:", lp.LpStatus[prob.status])
print("x1 =", lp.value(x1))
print("x2 =", lp.value(x2))
print("Objective value (Max):", max_obj_value)
print()

print("Objective 2 - Minimize x1 + 2x2:")
print("Status:", lp.LpStatus[prob.status])
print("x1 =", lp.value(x1))
print("x2 =", lp.value(x2))
print("Objective value (Min):", min_obj_value)
print()

# 用加权求和法寻找帕累托前线
weights = np.linspace(0, 1, 20)  # 在 0 和 1 之间生成 20 个均匀分布的权重
pareto_front = []

for w in weights:
    # 为每个权重组合创建一个新的 LP 问题
    prob_w = lp.LpProblem("Weighted Sum Problem", lp.LpMaximize)

    # 目标函数加权和
    prob_w += w * objective1 + (1 - w) * objective2

    # 增加制约因素
    prob_w += 0.5 * x1 + 0.25 * x2 <= 8
    prob_w += 0.2 * x1 + 0.2 * x2 <= 4
    prob_w += x1 + 5 * x2 <= 72
    prob_w += x1 + x2 >= 10

    # 解决问题
    prob_w.solve()

    # 存储找到的最佳值
    pareto_front.append((lp.value(objective1), lp.value(objective2)))

# 打印帕累托前沿解决方案
print("帕累托前沿解决方案 (目标1, 目标2):")
for solution in pareto_front:
    print(f"x1={solution[0]}, x2={solution[1]}")
