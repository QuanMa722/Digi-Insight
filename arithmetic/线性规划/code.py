# -*- coding: utf-8 -*-

from ortools.linear_solver import pywraplp
import pulp
import os


def LinerP(equation, austerity):

    expression = equation[0]
    lhs, rhs = expression.split("=")
    lhs = lhs.strip()
    rhs = rhs.strip()

    if lhs == "max":
        lp_problem = pulp.LpProblem("MyProbLP", sense=pulp.LpMaximize)
    else:
        lp_problem = pulp.LpProblem("MyProbLP", sense=pulp.LpMinimize)

    variables = {}
    for index in range(len(austerity)):
        if austerity[index][1] == "":
            if austerity[index][2] == "":
                variables[austerity[index][0]] = pulp.LpVariable(austerity[index][0],
                                                                 lowBound=None,
                                                                 upBound=None,
                                                                 cat=austerity[index][3])
            else:
                variables[austerity[index][0]] = pulp.LpVariable(austerity[index][0],
                                                                 lowBound=None,
                                                                 upBound=float(austerity[index][2]),
                                                                 cat=austerity[index][3])
        elif austerity[index][2] == "":
            variables[austerity[index][0]] = pulp.LpVariable(austerity[index][0],
                                                             lowBound=float(austerity[index][1]),
                                                             upBound=None,
                                                             cat=austerity[index][3])
        else:
            variables[austerity[index][0]] = pulp.LpVariable(austerity[index][0],
                                                             lowBound=float(austerity[index][1]),
                                                             upBound=float(austerity[index][2]),
                                                             cat=austerity[index][3])

    objective = eval(rhs, variables)
    lp_problem += objective

    for index in range(1, len(equation)):
        lp_problem += eval(equation[index], variables)

    original_stdout = os.dup(1)
    temp_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(temp_fd, 1)

    try:
        lp_problem.solve()

    except Exception as e:
        print(
            f"This is an error {e}, but the most likely is an input error."
        )

    os.dup2(original_stdout, 1)
    os.close(original_stdout)

    print("-" * 20)
    print(f"Result:{pulp.LpStatus[lp_problem.status]}")
    for var in lp_problem.variables():
        print(f"{var.name}:{var.varValue}")

    print(f"Objective:{pulp.value(lp_problem.objective)}")
    print("-" * 20)


def AssignP(task):
    num_workers = len(task)
    num_tasks = len(task[0])

    # Solver
    solver = pywraplp.Solver.CreateSolver("SCIP")

    if not solver:
        return

    x = {}
    for i in range(num_workers):
        for j in range(num_tasks):
            x[i, j] = solver.IntVar(0, 1, "")

    # Constraints
    # Each worker is assigned to exactly one task.
    for i in range(num_workers):
        solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 1)

    # Each task is assigned to exactly one worker.
    for j in range(num_tasks):
        solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)

    # Objective
    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(task[i][j] * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))

    # Solve
    status = solver.Solve()

    # Print solution.
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:

        print("-" * 20)
        print(f"Total cost = {solver.Objective().Value()}")
        print("-" * 20)

        for i in range(num_workers):
            for j in range(num_tasks):
                if x[i, j].solution_value() > 0.5:
                    print(f"Worker{i + 1} assigned to task{j + 1}." + f" Cost: {task[i][j]}")
    else:
        print("No solution found.")


if __name__ == '__main__':
    equation = [
        "max = 600 * x1 + 800 * x2 + 500 * x3 + 400 * x4 + 300 * x5",
        "x1 + x2 >= 20",
        "2000 * x1 + 4000 * x2 + 3000 * x3 + 5000 * x4 + 600 * x5 >= 100000",
        "1000 * x1 + 2000 * x2 <= 30000",
        "1000 * x1 + 2000 * x2 + 400 * x3 + 1000 * x4 + 100 * x5 <= 40000",
    ]

    # [variable, left border, right border, Integer/Continuous]

    austerity = [
        ['x1', 0, 14, 'Integer'],
        ['x2', 0, 8, 'Integer'],
        ['x3', 0, 40, 'Integer'],
        ['x4', 0, 5, 'Integer'],
        ['x5', 0, 50, 'Integer'],
    ]

    LinerP(equation, austerity)

    # Normal assignment, four jobs for four people.
    assign_n = [
        [90, 80, 75, 70],
        [35, 85, 55, 65],
        [125, 95, 90, 95],
        [45, 110, 95, 115],
    ]

    # Irregular assignment, four jobs for five people.
    assign_i = [
        [90, 80, 75, 70],
        [35, 85, 55, 65],
        [125, 95, 90, 95],
        [45, 110, 95, 115],
        [100, 85, 45, 88],
    ]

    AssignP(assign_n)
    AssignP(assign_i)
