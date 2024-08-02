# -*- coding: utf-8 -*-
# 状态随机，下一阶段的状态只与当前有关
# 以穿越沙漠为例

import numpy as np
import random

# 初始转移矩阵
transform_matrix = np.array(
    [
        [0.2222, 0.5556, 0.2222],
        [0.3571, 0.4286, 0.2143],
        [0.3333, 0.5000, 0.1667],
    ]
)

# 求解极限分布
AT = transform_matrix.T
eigenvalues, eigenvectors = np.linalg.eig(AT)
index = np.where(np.isclose(eigenvalues, 1))[0][0]
extreme_distribution = np.abs(eigenvectors[:, index])
extreme_distribution /= np.sum(extreme_distribution)

print("极限分布:", extreme_distribution)


def predict_daily_distribution(transform_matrix, initial_distribution, n_days):
    current_distribution = initial_distribution.copy()
    daily_distributions = [current_distribution]

    for _ in range(n_days - 1):
        current_distribution = np.dot(transform_matrix, current_distribution)
        # 生成每天的随机数，模拟实际情况
        random_numbers = [random.random() for _ in range(len(current_distribution))]
        # 归一化随机数，使其成为概率分布
        random_distribution = np.array(random_numbers) / sum(random_numbers)
        # 将状态分布与随机数混合，生成每天的实际预测状态分布
        mixed_distribution = 0.9 * current_distribution + 0.1 * random_distribution
        mixed_distribution /= np.sum(mixed_distribution)  # 归一化混合后的分布
        daily_distributions.append(mixed_distribution)

    return daily_distributions


# 预测未来10天的每天状态分布
n_days = 30
daily_distributions = predict_daily_distribution(transform_matrix, extreme_distribution, n_days)
for day, distribution in enumerate(daily_distributions):
    print("-" * 55)
    print(f"第{day + 1}天的预测状态分布:", distribution)
    print(f"第{day + 1}天预测出现A的概率:", distribution[0])
    print(f"第{day + 1}天预测出现B的概率:", distribution[1])
    print(f"第{day + 1}天预测出现C的概率:", distribution[2])
    print(f"第{day + 1}天的预测天气为:", ["A", "B", "C"][list(distribution).index(max(distribution))])
    print("-" * 55)
    print()

