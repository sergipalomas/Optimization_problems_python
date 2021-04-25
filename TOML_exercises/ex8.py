from cvxpy import *
import time
import numpy as np

def run():
    print('\n Solving exercise 8\n')

    # Create two scalar optimization variables.
    x1 = Variable()
    x2 = Variable()
    x3 = Variable()
    R12 = Variable()
    R23 = Variable()
    R32 = Variable()
    obj = Maximize(log(x1) + log(x2) + log(x3))
    constraints = [
        x1 + x2 <= R12,
        x1 <= R23,
        x3 <= R32,
        R12 + R23 + R32 <= 1
    ]

    # Form and solve problem.
    prob = Problem(obj, constraints)
    ts = time.time()
    prob.solve()  # Returns the optimal value.
    print("Time to converge = ", time.time() - ts)
    print("status:", prob.status)
    print("optimal value p* = ", prob.value)
    print("optimal var: x1 = ", x1.value, " x2 = ", x2.value, " x3 = ", x3.value)
    print("optimal var: R12 = ", R12.value, " R23 = ", R23.value, " R32 = ", R32.value)
    print("optimal dual variables λ1", constraints[0].dual_value)
    print("optimal dual variables λ2 = ", constraints[1].dual_value)
    print("optimal dual variables λ3 = ", constraints[2].dual_value)
    print("optimal dual variables λ4 = ", constraints[3].dual_value)
