from scipy.optimize import minimize
from autograd import grad, jacobian
import numpy as np
import matplotlib.pyplot as plt
import math
import time


print("Solving Exercise 1\n")

# Objective function
def obj_f(x):
    x1 = x[0]
    x2 = x[1]
    return math.exp(x1) * (4*x1**2 + 2*x2**2 + 4*x1*x2 + 2*x2 + 1)

# constraint functions
def ineq_constraint1(x):
    x1 = x[0]
    x2 = x[1]
    return -x1*x2 + x1 + x2 - 1.5

def ineq_constraint2(x):
    x1 = x[0]
    x2 = x[1]
    return -x1*x2 + 10

# Bounds
bounds_x1 = (None, None)
bounds_x2 = (None, None)

bound = [bounds_x1, bounds_x2]

constraint1 = {'type': 'ineq', 'fun': ineq_constraint1}
constraint2 = {'type': 'ineq', 'fun': ineq_constraint2}

constraint = [constraint1, constraint2]

def callbackF(Xi):
    global Nfeval, iterations, intermediate_values
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}'.format(Nfeval, Xi[0], Xi[1], obj_f(Xi)))
    iterations.append(Nfeval)
    intermediate_values.append(obj_f(Xi))
    Nfeval += 1

for x0 in ([0, 0], [10, 20], [-10, 1], [-30, -30]):
    Nfeval = 1
    iterations = list()
    intermediate_values = list()
    print("Inigial guess is:", x0)
    print('{0:4s}   {1:9s}   {2:9s}   {3:9s}'.format('Iter', ' X1', ' X2', 'f(X)'))
    ig = np.asarray(x0, dtype=np.float32)
    start_t = time.time()
    res = minimize(obj_f, ig, method='SLSQP', bounds=bound, constraints=constraint, callback=callbackF, options={'disp': True})
    end_t = time.time()
    plt.plot(iterations, intermediate_values)
    plt.xlabel('Iter')
    plt.ylabel('f(x)')
    #plt.show()
    print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1])
    print("Time to convergence: ", end_t-start_t)
    print("\n-----------------------\n\n")



# With Jacobian
constraint1 = {'type': 'ineq', 'fun': ineq_constraint1,
               'jac': lambda x: np.array([1-x[1], 1-x[0]])}
constraint2 = {'type': 'ineq', 'fun': ineq_constraint2,
               'jac': lambda x: np.array([-x[1], -x[0]])}
constraint = [constraint1, constraint2]

#obj_f_d = lambda x: np.array([(2*(2*x[0]*(x[0] + x[1] + 2) + x[1]*(x[1] + 3)) + 1)*math.exp(x[0]),
#                              math.exp(x[0])*(4*x[0] + 4*x[1] + 2)])
obj_f_d = jacobian(obj_f)

res = minimize(obj_f, ig, method='SLSQP', bounds=bound, jac=obj_f_d, constraints=constraint, options={'disp': True})
print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1])
print("Time to convergence: ", end_t - start_t)
print("\n-----------------------\n\n")