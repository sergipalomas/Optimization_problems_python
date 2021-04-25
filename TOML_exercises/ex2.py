from scipy.optimize import minimize
import numpy as np
import time
from autograd import jacobian, hessian, grad, elementwise_grad
import matplotlib.pyplot as plt

def run():
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


    # Plot cost function domain
    x1 = np.linspace(-50, 50, 101)
    x2 = np.linspace(-50, 50, 101)
    y = obj_f([x1, x2])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', xlabel='x1', ylabel='x2', zlabel='cost_func')
    ax.plot(x1, x2, y)
    plt.show()

    # Hessian matrix
    x_values = np.array([0., 0.], dtype=float)
    grad_cost = grad(obj_f)
    print(grad_cost(x_values))
    H_f = jacobian(elementwise_grad(obj_f))
    print("Hessian: \n", H_f(x_values))

    list_inigial_guess = ([[2, 2], [0, 2]])

    # Minimize
    for x0 in list_inigial_guess:
        print("Initial guess is:", x0)
        ig = np.asarray(x0, dtype=np.float32)
        start_t = time.time()
        res = minimize(obj_f, ig, method='SLSQP', bounds=bnds, constraints=cons, options={'disp': True})
        end_t = time.time()
        print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1])
        print("Time to convergence: ", end_t-start_t)
        print("\n-----------------------\n\n")

    print("Using Jacobian:")
    ig = np.array([2, 2], dtype=np.float32)
    print("Initial guess is:", ig)
    start_t = time.time()
    res2 = minimize(obj_f, ig, method='SLSQP', bounds=bnds, constraints=cons, jac=fun_Jac, options={'disp': True})
    end_t = time.time()
    print("optimal var: x1 = ", res2.x[0], " x2 = ", res2.x[1])
    print("Time to convergence: ", end_t - start_t)
    print("\n-----------------------\n\n")


    print("Using Jacobian:")
    ig = np.array([0, 2], dtype=np.float32)
    print("Initial guess is:", ig)
    start_t = time.time()
    res2 = minimize(obj_f, ig, method='SLSQP', bounds=bnds, constraints=cons, jac=fun_Jac, options={'disp': True})
    end_t = time.time()
    print("optimal var: x1 = ", res2.x[0], " x2 = ", res2.x[1])
    print("Time to convergence: ", end_t - start_t)
    print("\n-----------------------\n\n")