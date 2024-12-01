# -*- coding: utf-8 -*-

import pulp

# Specify CBC solver path if needed
# pulp.PULP_CBC_CMD(path='path_to_your_CBC_executable_here')

# Define the LP problem setup
prob = pulp.LpProblem("MyProbLP", sense=pulp.LpMaximize)

# Decision variables (corrected variable names)
# [variable, left border, right border, Integer/Continuous]
x1 = pulp.LpVariable('x1', lowBound=0, upBound=14, cat='Integer')
x2 = pulp.LpVariable('x2', lowBound=0, upBound=8, cat='Integer')
x3 = pulp.LpVariable('x3', lowBound=0, upBound=40, cat='Integer')
x4 = pulp.LpVariable('x4', lowBound=0, upBound=5, cat='Integer')
x5 = pulp.LpVariable('x5', lowBound=0, upBound=50, cat='Integer')

# Objective function
objective = 600 * x1 + 800 * x2 + 500 * x3 + 400 * x4 + 300 * x5

# Constraints
prob += x1 + x2 >= 20
prob += 2000 * x1 + 4000 * x2 + 3000 * x3 + 5000 * x4 + 600 * x5 >= 100000
prob += 1000 * x1 + 2000 * x2 <= 30000
prob += 1000 * x1 + 2000 * x2 + 400 * x3 + 1000 * x4 + 100 * x5 <= 40000

# Solve the problem
prob.solve()

# Print results
print("Status:", pulp.LpStatus[prob.status])
print("Objective Value:", pulp.value(objective))
for v in prob.variables():
    print(v.name, "=", v.varValue)
