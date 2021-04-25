from scipy.optimize import minimize
import matplotlib.pyplot as plt
from math import e
import time
import autograd.numpy as np
from autograd import grad, jacobian, elementwise_grad

print("Solving Exercise 1\n")

# Objective function
def obj_f(x):
    x1 = x[0]
    x2 = x[1]
    return e**(x1) * (4*x1**2 + 2*x2**2 + 4*x1*x2 + 2*x2 + 1)

# Jacobian
def fun_Jac(x):
    x1 = x[0]
    x2 = x[1]
    dx1 = e**x1 * (1 + 4*x1**2 + 6*x2 + 2*x2**2 + 4*x1*(2 + x2))
    dx2 = e**x1 * (2 + 4*x1 + 4*x2)
    return np.array((dx1, dx2))

# constraint functions
def ineq_constraint1(x):
    x1 = x[0]
    x2 = x[1]
    return -x1*x2 + x1 + x2 - 1.5

def ineq_constraint2(x):
    x1 = x[0]
    x2 = x[1]
    return x1*x2 + 10

def callbackF(Xi):
    global Nfeval, iterations, intermediate_values
    if (round(Xi[0] * Xi[1] - Xi[0] - Xi[1], 4) <= -1.5) and (round(-Xi[0] * Xi[1], 4) <= 10):
        feasible = True
    else: feasible = False
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}    {4: 1d}'.format(Nfeval, Xi[0], Xi[1], obj_f(Xi), feasible))
    iterations.append(Nfeval)
    intermediate_values.append(obj_f(Xi))
    Nfeval += 1

# CHECK CONVEXITY

# Plot cost function domain
x1 = np.linspace(-30, 30, 200)
x2 = np.linspace(-30, 30, 200)
X1, X2 = np.meshgrid(x1, x2)
C = e**X1*(4*X1**2 + 2*X2**2 + 4*X1*X2 + 2*X2 + 1)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, C, rstride=1, cstride=1, cmap='cividis', edgecolor='none')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('cost_function')
ax.view_init(30, -240)
plt.show()

# Hessian matrix
x_values = np.array([0., 0.], dtype=float)
grad_cost = grad(obj_f)
print(grad_cost(x_values))
H_f = jacobian(elementwise_grad(obj_f))
print("Hessian: \n", H_f(x_values))


# Bounds
bounds_x1 = (None, None)
bounds_x2 = (None, None)

bound = [bounds_x1, bounds_x2]

constraint1 = {'type': 'ineq', 'fun': ineq_constraint1}
constraint2 = {'type': 'ineq', 'fun': ineq_constraint2}

constraint = [constraint1, constraint2]

for x0 in ([10, 20], [-10, 1], [-30, -30.], [0, 0]):
    Nfeval = 1
    iterations = list()
    intermediate_values = list()
    print("Inigial guess is:", x0)
    print("Constraint 1: ", round(x0[0]*x0[1] - x0[0] - x0[1], 4), " <= -1.5")
    print("Constraint 2.", round(-x0[0]*x0[1], 4), "<= 10")
    if (round(x0[0]*x0[1] - x0[0] - x0[1], 4) <= -1.5) and (round(-x0[0]*x0[1], 4) <= 10):
        print("Constraints satisfied in Initial Guess")
    else: print("Initial Guess is not feasible!")
    print('{0:4s}   {1:9s}   {2:9s}   {3:9s}    {4:5s}'.format('Iter', ' X1', ' X2', 'f(X)', 'Feasible'))
    ig = np.asarray(x0, dtype=np.float32)
    start_t = time.time()
    res = minimize(obj_f, ig, method='SLSQP', bounds=bound, constraints=constraint, callback=callbackF, options={'disp': True})
    end_t = time.time()
    plt.plot(iterations, intermediate_values)
    title = "Initial guess: " + np.array2string(ig)
    plt.title(title)
    plt.xlabel('Iter')
    plt.ylabel('f(x)')
    plt.show()
    print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1])
    print("Time to convergence: ", end_t-start_t)
    print("Constraint 1: ", round(res.x[0] * res.x[1] - res.x[0] - res.x[1], 4), " <= -1.5")
    print("Constraint 2.", round(-res.x[0] * res.x[1], 4), "<= 10")
    if (round(res.x[0] * res.x[1] - res.x[0] - res.x[1], 4) <= -1.5) and (round(-res.x[0] * res.x[1], 4) <= 10):
        print("Result is feasible")
    else: print("Result NOT feasible")
    print("\n-----------------------\n\n")



# With Jacobian
constraint1 = {'type': 'ineq', 'fun': ineq_constraint1}
constraint2 = {'type': 'ineq', 'fun': ineq_constraint2}
constraint = [constraint1, constraint2]
J_f = jacobian(obj_f)
print("Using Jacobian")
ig = np.asarray((0,0), dtype=np.float32)
print("Inigial guess is:", ig)
res = minimize(obj_f, ig, method='SLSQP', bounds=bound, jac=fun_Jac, constraints=constraint, options={'disp': True})
print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1])
print("Time to convergence: ", end_t - start_t)
print("\n-----------------------\n\n")
