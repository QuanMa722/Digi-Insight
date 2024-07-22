# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pulp
import os


class Operation:

    @staticmethod
    def liner_program(equation, austerity):

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
                print("austerity error")

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

        print(pulp.LpStatus[lp_problem.status])
        for var in lp_problem.variables():
            print(f"{var.name}: {var.varValue}")
        print(f"Objective: {pulp.value(lp_problem.objective)}")


class Statistic:

    def __init__(self, file):
        self.file = file

    def average(self, column):
        try:
            if isinstance(column, list):
                column_array = np.array(column)
                average_value = np.mean(column_array)
                return average_value
            else:
                try:
                    data = pd.read_csv(self.file, encoding="utf-8")
                except UnicodeDecodeError:
                    data = pd.read_csv(self.file, encoding="gbk")
                except FileNotFoundError:
                    print("File not found")
                    return

                column_array = np.array(data[column])
                average_value = np.mean(column_array)
                return average_value

        except Exception as e:
            print(f"An error occurred: {e}")

    def variance(self, column):
        try:
            if isinstance(column, list):
                column_array = np.array(column)
                variance_value = np.var(column_array)
                return variance_value
            else:
                try:
                    data = pd.read_csv(self.file, encoding="utf-8")
                except UnicodeDecodeError:
                    data = pd.read_csv(self.file, encoding="gbk")
                except FileNotFoundError:
                    print("File not found")
                    return

                column_array = np.array(data[column])
                variance_value = np.var(column_array)
                return variance_value

        except Exception as e:
            print(f"An error occurred: {e}")

    def max(self, column):
        try:
            if isinstance(column, list):
                max_value = max(column)
                return max_value
            else:
                try:
                    data = pd.read_csv(self.file, encoding="utf-8")
                except UnicodeDecodeError:
                    data = pd.read_csv(self.file, encoding="gbk")
                except FileNotFoundError:
                    print("File not found")
                    return

                column_array = np.array(data[column])
                max_value = np.max(column_array)
                return max_value

        except Exception as e:
            print(f"An error occurred: {e}")

    def min(self, column):
        try:
            if isinstance(column, list):
                min_value = min(column)
                return min_value
            else:
                try:
                    data = pd.read_csv(self.file, encoding="utf-8")
                except UnicodeDecodeError:
                    data = pd.read_csv(self.file, encoding="gbk")
                except FileNotFoundError:
                    print("File not found")
                    return

                column_array = np.array(data[column])
                min_value = np.min(column_array)
                return min_value

        except Exception as e:
            print(f"An error occurred: {e}")

    def median(self, column):
        try:
            if isinstance(column, list):
                median_value = np.median(column)
                return median_value
            else:
                try:
                    data = pd.read_csv(self.file, encoding="utf-8")
                except UnicodeDecodeError:
                    data = pd.read_csv(self.file, encoding="gbk")
                except FileNotFoundError:
                    print("File not found")
                    return

                column_array = np.array(data[column])
                median_value = np.median(column_array)
                return median_value

        except Exception as e:
            print(f"An error occurred: {e}")


class MachineLearning:
    pass
