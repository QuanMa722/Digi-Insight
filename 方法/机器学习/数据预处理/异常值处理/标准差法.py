# -*- coding: utf-8 -*-

import pandas as pd

# 示例数据
data = {
    'Values': [10, 12, 14, 15, 16, 18, 20, 22, 24, 25, 30, 32, 40, 100]  # 这里的100是一个异常值
}

df = pd.DataFrame(data)

# 计算均值和标准差
mean = df['Values'].mean()
std_dev = df['Values'].std()

# 设置阈值，例如3个标准差
threshold = 3

# 确定异常值的上下界
lower_bound = mean - threshold * std_dev
upper_bound = mean + threshold * std_dev

# 标记异常值
df['Outlier'] = df['Values'].apply(lambda x: x < lower_bound or x > upper_bound)

# 显示计算结果
print(f"均值: {mean}")
print(f"标准差: {std_dev}")
print(f"异常值上下界限: ({lower_bound}, {upper_bound})")
print("数据及其异常值标记:")
print(df)



