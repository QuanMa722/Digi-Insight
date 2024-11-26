# -*- coding: utf-8 -*-

import pandas as pd

# 示例 DataFrame
df = pd.DataFrame({
    'A': [1, None, 3, None],
    'B': ['a', 'b', None, 'd']
})

# 删除空行
df_cleaned = df.dropna()

print(df_cleaned)
