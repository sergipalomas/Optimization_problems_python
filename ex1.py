from scipy.optimize import minimize
import numpy as np
import math

def solve():
    print("Solving Exercise 1\n")

    # Objective function
    fun = lambda x: math.exp(x[0]) * 4*x[0]**2 + 2 * x[1]**2 + 4*x[0]*x[1] + 2*x[1] + 1

    # constraints functions
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] * x[1] - x[0] - x[1] - 1.5},
            {'type': 'ineq', 'fun': lambda x: -x[0] * x[1] + 10})

    bnds = ((None, None), )*2

    for x0 in ([0,0],[10,20],[-10,1],[-30,-30]):
        print("Inigial guess is:", x0)
        ig = np.asarray(x0, dtype=np.float32)
        res = minimize(fun, ig, method='SLSQP', bounds=bnds, constraints=cons)
        print (res)
        print ("optimal value p*", res.fun)
        print ("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1], "\n-----------------------\n\n")

