from cvxopt import *
import time
import numpy as np
import matplotlib.pyplot as plt

def run():
    f = lambda x: 2*x**2 - 0.5
    df = lambda x: 4*x

    ## For ex6 part 2:
    # f = lambda x: 2*x**4 - 4*x**2 + x - 0.5
    # df = lambda x: 8*x**3 - 8*x + 1

    x0 = 2
    x = x0
    grad = df(x0)  # df/dx = 4*x

    # Search direction
    p = -grad

    # Stopping criteria
    learning_rate = 10**-4

    alpha = 0.01  # 0 <= alpha <= 0.5
    beta = 0.8  # 0 <= beta <= 1

    points = list()
    points.append(x0)

    cnt = 1

    while np.linalg.norm(np.asarray(grad)) > learning_rate:
        # Set t to 1 every time we enter the loop and calculate the new t
        t = 1
        t = backtrack(f, df, p, grad, x, t, alpha, beta, cnt)
        grad = df(x)
        p = -grad
        x = x + t*p
        points.append(x)


    ypoints = [f(x) for x in points]

    # Plot
    x1 = np.linspace(-2.2, 2.2, 50)
    y = f(x1)
    fig = plt.figure()
    ax = fig.add_subplot(xlabel='x', ylabel='cost_func')
    ax.plot(x1, y)
    ax.plot(points, ypoints)
    textstr = '\n'.join((
        r'x0 = %.2f' % (x0),
        r'$\alpha = %.2f$' % (alpha),
        r'$\beta$ = %.2f' % (beta)
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.5, 0.95, textstr, transform=ax.transAxes, verticalalignment='top', bbox=props)
    plt.show()

    print("\nfinal step size :",  t)
    print("list of poiints: ", points)
    print(ypoints)

def backtrack(f, df, p, grad, x, t, alpha, beta, count):
    while f(x + t*p) > f(x) + alpha * t * grad * p:
        t *= beta
        print("""

########################
###   iteration {}   ###
########################
""".format(count))
        print("Inequality: ",  f(x) - (f(x - t*np.array([df(x)])) + alpha * t * np.dot(np.array([df(x)]), np.array([df(x)]))))
        count += 1
    return t