# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

np.random.seed(0)
data = np.random.normal(loc=0, scale=1, size=100)
sm.qqplot(data, line='45', marker='o')

plt.title('QQ Plot')

plt.grid()
plt.tight_layout()
plt.show()
