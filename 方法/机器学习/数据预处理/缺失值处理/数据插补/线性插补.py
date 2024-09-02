# -*- coding: utf-8 -*-
"""
线性插值（或称为线性插补）是一种用于估算两个已知数据点之间未知值的方法。
这种方法假设在已知数据点之间的变化是线性的，也就是说，数据点之间的变化是均匀的。
其主要目的是用来填补或估算在已知数据点之间缺失的数据值。
"""

import pandas as pd
import numpy as np

# 示例数据：包含缺失值的 DataFrame
data = {
    'x': [1, 2, 4, 5],        # x 值
    'y': [10, np.nan, 30, 40]  # y 值，其中有一个缺失值
}

df = pd.DataFrame(data)

# 使用线性插值填补缺失值
df['y'] = df['y'].interpolate(method='linear')

print(df)
