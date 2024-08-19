# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = load_iris()
iris_target = data.target
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)
iris_all = iris_features.copy()

iris_all['target'] = iris_target
corr = iris_all.corr()

ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(corr, annot=True, cmap='Blues')

plt.show()
