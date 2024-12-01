# -*- coding: utf-8 -*-

from ortools.linear_solver import pywraplp


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
