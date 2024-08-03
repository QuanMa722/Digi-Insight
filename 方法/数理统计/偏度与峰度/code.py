# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def visual(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    # 绘制直方图
    sns.histplot(data, kde=True, ax=ax, color='skyblue', stat='density')
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    # 添加标签和标题
    ax.set_title(f'数据分布\n偏度：{skewness}  峰度：{kurtosis}')
    ax.set_xlabel('数据值')
    ax.set_ylabel('密度')
    # 显示图形
    plt.tight_layout()
    plt.show()


def statistics(data):
    # 偏度与峰度
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)

    # print(f"偏度：{skewness}\n峰度：{kurtosis}")

    print("-" * 20)
    if abs(skewness) <= 0.5:
        print("偏度符合正态分布")
    else:
        if skewness > 0:
            print("分布靠左，称为正偏或右偏")
        elif skewness < 0:
            print("分布靠右，称为负偏或左偏")
        else:
            print("分布形态与正态分布相同")

    # 超值峰度
    if abs(kurtosis) <= 0.5:
        print("峰度符合正态分布")
    else:
        if kurtosis > 0:
            print("分布偏向高而尖")
        elif kurtosis < 0:
            print("分布偏向扁而平")
        else:
            print("分布与正态分布相同")
    print("-" * 20)


if __name__ == '__main__':

    data = np.random.randn(10000)
    statistics(data)
    visual(data)
