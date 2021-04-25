from cvxpy import *
import time

def run():
    print('\n Solving exercise 4\n')

    # Create two scalar optimization variables.
    x = Variable()

    obj = Minimize(x**2+1)
    constraints = [
        x**2 - 6*x + 8 <= 0
    ]
    #constraints = [sum_squares((x+y)) <= 3,
    #               3*x + 2*y >= 3
    #               ]

    # Form and solve problem.
    prob = Problem(obj, constraints)
    ts = time.time()
    prob.solve()  # Returns the optimal value.
    print("time to converge (s) = ", time.time() - ts)
    print("status:", prob.status)
    print("optimal value p* = ", prob.value)
    print("optimal var: x1 = ", x.value)
    print("optimal dual variables lanbda1 = ", constraints[0].dual_value)