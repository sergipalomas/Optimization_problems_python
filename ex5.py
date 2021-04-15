from cvxpy import *
import numpy as np


print('\n Solving exercise 4\n')

# Create two scalar optimization variables.
x = Variable()
y = Variable()

obj = Minimize(x**2 + y**2)
constraints = [
    x**2 + y**2 - 2*x - 2*y + 2 <= 1,
    x**2 + y**2 - 2*x + 2*y + 2 <= 1
]

# Form and solve problem.
prob = Problem(obj, constraints)
print("solve", prob.solve())  # Returns the optimal value.
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x1 = ", x.value, " x2 = ", y.value)
print("optimal dual variables lanbda1 = ", constraints[0].dual_value)
print ("optimal dual variables lanbda2 = ", constraints[1].dual_value)