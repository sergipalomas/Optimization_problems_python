from scipy.optimize import minimize
import numpy as np
from autograd import jacobian

print("Solving Exercise 2\n")

# Objective function
obj_f = lambda x: x[0]**2 + x[1]**2

# Jacobian
def fun_Jac(x):
    dx = 2*x[0]
    dy = 2*x[1]
    return np.array((dx, dy))

# constraints functions
cons = ({'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 1.},
        {'type': 'ineq', 'fun': lambda x: x[0]**2 + x[1]**2 - 1.},
        {'type': 'ineq', 'fun': lambda x: 9*x[0]**2 + x[1]**2 - 9.},
        {'type': 'ineq', 'fun': lambda x: x[0]**2 - x[1]},
        {'type': 'ineq', 'fun': lambda x: -x[0] + x[1]**2})

# Bounds
bnds = ((0.5, None), (None, None))

list_inigial_guess = ([[1, 0], [0, 0], [1000, -2000]])

# Minimize
for x0 in list_inigial_guess:
    print("Initial guess is:", x0)
    ig = np.asarray(x0, dtype=np.float32)
    res = minimize(obj_f, ig, method='SLSQP', bounds=bnds, constraints=cons, options={'disp': True})
    print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1])
    print("\n-----------------------\n\n")

print("Using Jacobian:")
ig = np.array([10, 20], dtype=np.float32)
print("Initial guess is:", ig)
res2 = minimize(obj_f, ig, method='SLSQP', bounds=bnds, constraints=cons, jac=fun_Jac, options={'disp': True})
print("optimal var: x1 = ", res2.x[0], " x2 = ", res2.x[1])
print("\n-----------------------\n\n")
