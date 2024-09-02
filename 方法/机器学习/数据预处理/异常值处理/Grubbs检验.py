# -*- coding: utf-8 -*-
# Grubbs检验用于检测单个异常值，适用于正态分布的数据。

from scipy import stats
import numpy as np


def grubbs_test(data, alpha=0.05):
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)

    # 计算Grubbs统计量
    G_max = (np.max(data) - mean) / std_dev
    G_min = (mean - np.min(data)) / std_dev
    G = max(G_max, G_min)

    # 计算临界值
    critical_value = stats.t.ppf(1 - alpha / (2 * n), n - 2) / np.sqrt(n - 1)
    critical_value = np.sqrt((n - 1) / (n - 2 + critical_value ** 2))

    return G > critical_value, G


# 示例数据
data = [10, 12, 14, 15, 16, 18, 20, 22, 24, 25, 30, 32, 40, 100]

# 执行Grubbs检验
is_outlier, G_value = grubbs_test(data)

# 打印Grubbs统计量
print(f"Grubbs统计量: {G_value:.3f}")
print(f"是否存在异常值: {is_outlier}")

if is_outlier:
    # 找出并打印异常值
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    outliers = [x for x in data if (x > mean + 3 * std_dev) or (x < mean - 3 * std_dev)]

    print("数据中的异常值有：", outliers)
else:
    print("数据中没有异常值。")
