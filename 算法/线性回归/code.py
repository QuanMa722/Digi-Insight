# -*- coding: utf-8 -*-

from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

X = [100, 110, 120, 130, 140]
y = [100 * 1, 110 * 1.05, 120 * 1.1, 130 * 0.95, 140 * 0.9]

plt.ylim(80, 160)
plt.xlim(80, 160)
plt.scatter(X, y, c='k')
plt.plot(np.arange(80, 160), np.arange(80, 160) * 1 + 0, c='k')
plt.show()

model = linear_model.LinearRegression()
X = np.array(X).reshape(-1, 1)
model.fit(X, y)

# y = ax + b
a = model.coef_
b = model.intercept_

print("-" * 25)
print("a =", model.coef_[0])
print("b =", model.intercept_)
print("-" * 25)
