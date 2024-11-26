# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

equation_str: str = "x ** 6 - 5 * x ** 5 + 3 * x ** 4 + x ** 3 - 7 * x ** 2 + 7 * x - 20"


def f(x):
    equation = eval(equation_str)
    return equation


plt.figure(figsize=(8, 6))
x_vals = np.linspace(-2, 5, 400)
y_vals = f(x_vals)

plt.plot(x_vals, y_vals, label=f'f(x) = {equation_str}', color='black')
plt.title(f'f(x) = {equation_str}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()

