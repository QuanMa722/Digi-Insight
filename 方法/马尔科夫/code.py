# -*- coding: utf-8 -*-
# 状态随机，下一阶段的状态只与当前有关
# 以穿越沙漠为例

import numpy as np

# 初始转移矩阵
transform_matrix = np.array([
    [0.2222, 0.5556, 0.2222],
    [0.3571, 0.4286, 0.2143],
    [0.3333, 0.5000, 0.1667],
])


# 求解极限分布
def calculate_extreme_distribution(transform_matrix):
    AT = transform_matrix.T
    eigenvalues, eigenvectors = np.linalg.eig(AT)
    index = np.where(np.isclose(eigenvalues, 1))[0][0]
    extreme_distribution = np.abs(eigenvectors[:, index])
    return extreme_distribution / np.sum(extreme_distribution)


extreme_distribution = calculate_extreme_distribution(transform_matrix)
print("极限分布:", extreme_distribution)


def predict_daily_distribution(transform_matrix, initial_distribution, n_days, random_factor=0.1):
    daily_distributions = [initial_distribution.copy()]

    for _ in range(1, n_days):
        # 计算当前分布
        current_distribution = np.dot(transform_matrix, daily_distributions[-1])

        # 生成随机分布并归一化
        random_distribution = np.random.rand(len(current_distribution))
        random_distribution /= np.sum(random_distribution)

        # 混合当前分布和随机分布
        mixed_distribution = (1 - random_factor) * current_distribution + random_factor * random_distribution
        daily_distributions.append(mixed_distribution / np.sum(mixed_distribution))  # 归一化

    return daily_distributions


# 预测未来30天的每天状态分布
n_days = 10
daily_distributions = predict_daily_distribution(transform_matrix, extreme_distribution, n_days)

# 输出结果
for day, distribution in enumerate(daily_distributions):
    print("-" * 55)
    print(f"第{day + 1}天的预测状态分布:", distribution)
    print(f"第{day + 1}天预测出现A的概率:", distribution[0])
    print(f"第{day + 1}天预测出现B的概率:", distribution[1])
    print(f"第{day + 1}天预测出现C的概率:", distribution[2])
    print(f"第{day + 1}天的预测天气为:", ["A", "B", "C"][np.argmax(distribution)])
    print("-" * 55)
    print()
