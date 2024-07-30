# -*- coding: utf-8 -*-
# linear weighting (math.)

import numpy as np
import pulp as lp

# Define the LP problem setup (similar to previous example)
prob = lp.LpProblem("Multi-Objective Problem", lp.LpMaximize)

# Decision variables
x1 = lp.LpVariable('x1', lowBound=0, upBound=None, cat='Continuous')
x2 = lp.LpVariable('x2', lowBound=0, upBound=None, cat='Continuous')

# Objective functions
objective1 = 2 * x1 + 3 * x2
objective2 = x1 + 2 * x2

# Constraints (same as before)
prob += 0.5 * x1 + 0.25 * x2 <= 8
prob += 0.2 * x1 + 0.2 * x2 <= 4
prob += x1 + 5 * x2 <= 72
prob += x1 + x2 >= 10

# Solve the problem for each objective separately
prob.solve()

# Retrieve the optimal values for each objective
max_obj_value = lp.value(objective1)
min_obj_value = lp.value(objective2)

print("Objective 1 - Maximize 2x1 + 3x2:")
print("Status:", lp.LpStatus[prob.status])
print("x1 =", lp.value(x1))
print("x2 =", lp.value(x2))
print("Objective value (Max):", max_obj_value)
print()

print("Objective 2 - Minimize x1 + 2x2:")
print("Status:", lp.LpStatus[prob.status])
print("x1 =", lp.value(x1))
print("x2 =", lp.value(x2))
print("Objective value (Min):", min_obj_value)
print()

# Finding the Pareto Front using weighted-sum method
weights = np.linspace(0, 1, 20)  # Generate 20 evenly spaced weights between 0 and 1
pareto_front = []

for w in weights:
    # Create a new LP problem for each weight combination
    prob_w = lp.LpProblem("Weighted Sum Problem", lp.LpMaximize)

    # Objective function as a weighted sum
    prob_w += w * objective1 + (1 - w) * objective2

    # Add constraints (same as before)
    prob_w += 0.5 * x1 + 0.25 * x2 <= 8
    prob_w += 0.2 * x1 + 0.2 * x2 <= 4
    prob_w += x1 + 5 * x2 <= 72
    prob_w += x1 + x2 >= 10

    # Solve the problem
    prob_w.solve()

    # Store the optimal values found
    pareto_front.append((lp.value(objective1), lp.value(objective2)))

# Print the Pareto front solutions
print("Pareto Front Solutions (Objective 1, Objective 2):")
for solution in pareto_front:
    print(f"x1={solution[0]}, x2={solution[1]}")
