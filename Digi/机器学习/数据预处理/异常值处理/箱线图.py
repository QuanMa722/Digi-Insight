# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置字体以支持中文
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 示例数据
data = {
    'Values': [10, 12, 14, 15, 16, 18, 20, 22, 24, 25, 30, 32, 40, 100]  # 这里的100是一个异常值
}
df = pd.DataFrame(data)

# 计算四分位数
Q1 = df['Values'].quantile(0.25)
Q3 = df['Values'].quantile(0.75)
IQR = Q3 - Q1

# 确定异常值的上下界
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 标记异常值
df['Outlier'] = df['Values'].apply(lambda x: x < lower_bound or x > upper_bound)

# 创建箱线图，不显示原异常值点
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x=df['Values'], linewidth=2, showfliers=False)

# 突出显示异常值
outliers = df[df['Outlier']]
plt.scatter(outliers['Values'], [0] * len(outliers), color='#b83b5e', s=100, label='异常值')

# 添加标题和标签
plt.title('箱线图示例', fontsize=14)
plt.xlabel('值', fontsize=12)

# 显示图例
plt.legend(fontsize=10)

# 显示图形
plt.show()
