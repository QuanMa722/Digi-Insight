# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = {
    'Class': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C'],
    'Score': [85, 90, 88, 75, 95, 70, 82, 78, 88, 98, 60, 65, 77, 80, 72]
}

df = pd.DataFrame(data)

plt.figure(figsize=(8, 6))
sns.boxplot(x='Class', y='Score', data=df)
plt.title('Boxplot of Scores by Class')
plt.xlabel('Class')
plt.ylabel('Score')
plt.grid(True)
plt.show()
