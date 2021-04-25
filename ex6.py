from cvxopt import *
import time
import numpy as np

f = lambda x: (2*x**2 - 0.5)

def dfx1(x):
    return 4*x

t = 1
cnt = 1
x0 = 3
alpha = 0.3  # 0 <= alpha <= 0.5
beta = 0.8  # 0 <= beta <= 1
learning_rate = 10**-4

def backtrack(x0, dfx1, t, alpha, beta, count):
    while (f(x0) - (f(x0 - t*np.array([dfx1(x0)])) + alpha * t * np.dot(np.array([dfx1(x0)]), np.array([dfx1(x0)])))) < 0:
        t *= beta
        print("""

########################
###   iteration {}   ###
########################
""".format(count))
        print("Inequality: ",  f(x0) - (f(x0 - t*np.array([dfx1(x0)])) + alpha * t * np.dot(np.array([dfx1(x0)]), np.array([dfx1(x0)]))))
        count += 1
    return t

t = backtrack(x0, dfx1, t, alpha, beta, cnt)

print("\nfinal step size :",  t)
