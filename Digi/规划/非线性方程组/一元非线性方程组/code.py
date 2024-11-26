# -*- coding: utf-8 -*-
# Brent‘s method

from scipy.optimize import brentq


def f(x):
    return x ** 6 - 5 * x ** 5 + 3 * x ** 4 + x ** 3 - 7 * x ** 2 + 7 * x - 20


a = 4
b = 5

root = brentq(f, a, b, xtol=1e-12)
print("Approximate root:", root)

