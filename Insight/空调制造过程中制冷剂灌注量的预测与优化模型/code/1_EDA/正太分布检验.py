# -*- coding: utf-8 -*-

from scipy import stats
import pandas as pd

# 读取数据
data = pd.read_excel("data1.xlsx")

# 设定显著性水平
alpha = 0.1

# 检验列
columns_to_check = ['f1', 'f2', 'f4', 'f5', 'f6', 'f7', 'f8', 'f10', 'f11', 'f12']

# Anderson-Darling 正态性检验
for col in columns_to_check:
    # 提取当前列的数据
    sample_data = data[col].dropna().values  # 去除 NaN 数据

    # 执行 Anderson-Darling 检验，检验数据是否符合正态分布
    result = stats.anderson(sample_data, dist='norm')

    # 选择对应显著性水平的临界值（根据 alpha 选择临界值的索引位置）
    critical_value = result.critical_values[2]  # 索引2是对应 alpha=0.1 的临界值

    # 判断统计量与临界值的关系
    if result.statistic > critical_value:
        # 如果统计量大于临界值，拒绝正态分布假设
        # print('-' * 20)
        # print(f"变量 {col} 不服从正态分布。")
        # print(f"检验统计量: {result.statistic:.4f}, 临界值: {critical_value:.4f}")
        # print('-' * 20)
        pass

    else:
        # 否则，接受正态分布假设
        print(f"变量 {col} 服从正态分布。")
        print(f"检验统计量: {result.statistic:.4f}, 临界值: {critical_value:.4f}")
        print()
