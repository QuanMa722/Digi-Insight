# -*- coding: utf-8 -*-

import pandas as pd

# 设置 Pandas 显示最大列数
pd.set_option('display.max_columns', None)  # 不限制列数
pd.set_option('display.width', None)  # 不限制宽度

data = pd.read_excel('data1.xlsx')

# 数据前五行
print(data.head())
print()

# 数据统计
print(data.describe())