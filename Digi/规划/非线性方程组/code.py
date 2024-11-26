# -*- coding: utf-8 -*-

from math import *
from sympy import *
import numpy as np
import sympy


class Newton:
    def __init__(self, equations):
        self.equations = equations
        self.n = len(equations)
        self.x = sympy.symbols("x0:{}".format(self.n), real=True)
        self.equationsSymbol = [equations[i](self.x) for i in range(self.n)]
        self.J = np.zeros(self.n * self.n, dtype=sympy.core.add.Add).reshape(self.n, self.n)
        for i in range(self.n):
            for j in range(self.n):
                self.J[i][j] = sympy.diff(self.equationsSymbol[i], self.x[j])

    def cal_J(self, x):
        dict = {self.x[i]: x[i] for i in range(self.n)}
        J = np.zeros(self.n * self.n).reshape(self.n, self.n)
        for i in range(self.n):
            for j in range(self.n):
                J[i][j] = self.J[i][j].subs(dict)
        return J

    def cal_f(self, x):
        f = np.zeros(self.n)
        for i in range(self.n):
            f[i] = self.equations[i](x)
        f.reshape(self.n, 1)
        return f

    def byStep(self, x0, step):
        x0 = np.array(x0)
        for i in range(step):
            x0 = x0 - np.linalg.pinv(self.cal_J(x0)) @ self.cal_f(x0)
            print("Step {}:".format(i + 1), ", ".join(["x{} = {}".format(j + 1, x0[j]) for j in range(self.n)]))
        return x0

    def byEpsilon(self, x0, epsilon):
        error = float("inf")
        while error >= epsilon:
            cal = np.linalg.pinv(self.cal_J(x0)) @ self.cal_f(x0)
            error = max(abs(cal))
            x0 = x0 - cal
        print("-" * 60)
        print("Approximate root:", x0)
        print("-" * 60)
        return x0


if __name__ == "__main__":
    # equations 为 equation = 0 的方程组
    # byStep(x0, step): x0 为初始值 step 为迭代次数
    # byEpsilon(x0, epsilon): x0 为初始值 epsilon 为精度

    # 多元非线性方程组
    equations = [
        lambda x: cos(0.4 * x[1] + x[0] ** 2) + x[0] ** 2 + x[1] ** 2 - 1.6,
        lambda x: 1.5 * x[0] ** 2 - x[1] ** 2 / 0.36 - 1,
        lambda x: 3 * x[0] + 4 * x[1] + 5 * x[2]
    ]

    newton = Newton(equations)
    newton.byStep([1, 1, 1], 5)
    newton.byEpsilon([1, 1, 1], 1e-12)

    # 一元非线性方程组
    equations = [
        lambda x: x[0] ** 6 - 5 * x[0] ** 5 + 3 * x[0] ** 4 + x[0] ** 3 - 7 * x[0] ** 2 + 7 * x[0] - 20
    ]

    newton = Newton(equations)
    newton.byStep([4], 5)
    newton.byEpsilon([4], 1e-12)
