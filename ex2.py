from scipy.optimize import minimize
import numpy as np
from autograd import jacobian

print("Solving Exercise 2\n")

# Objective function
obj_f = lambda x: x[0]**2 + x[1]**2

# constraints functions
cons = ({'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 1.},
        {'type': 'ineq', 'fun': lambda x: x[0]**2 + x[1]**2 - 1.},
        {'type': 'ineq', 'fun': lambda x: 9*x[0]**2 + x[1]**2 - 9.},
        {'type': 'ineq', 'fun': lambda x: x[0]**2 - x[1]},
        {'type': 'ineq', 'fun': lambda x: -x[0] + x[1]**2})

# Bounds
bnds = ((0.5, None), (None, None))

# Minimize
for x0 in ([1, 0], [0, 0]):
    print("Inigial guess is:", x0)
    ig = np.asarray(x0, dtype=np.float32)
    res = minimize(obj_f, ig, method='SLSQP', bounds=bnds, constraints=cons)
    print(res)
    print("optimal value p*", res.fun)
    print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1], "\n-----------------------\n\n")

