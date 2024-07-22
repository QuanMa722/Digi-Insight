# -*- coding: utf-8 -*-

from sorml import Operation

equation = [
    "max = 60 * x + 100 * y",
    "0.18 * x + 0.09 * y <= 72",
    "0.08 * x + 0.28 * y <= 56",
]

austerity = [
    ['x', 0, '', 'Continuous'],
    ['y', 0, '', 'Continuous'],
]

liner_program = Operation()
liner_program.liner_program(equation, austerity)