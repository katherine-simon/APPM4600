import numpy as np
import math
from tabulate import tabulate

def driver():
    f = lambda x: x[0]**2 + 4*x[1]**2 + 4*x[2]**2 - 16
    fx = lambda x: 2*x[0]
    fy = lambda x: 8*x[1]
    fz = lambda x: 8*x[2]

    x0 = np.array([1,1,1])

    roots = []

    for n in range(100):
        x1 = np.zeros(3)
        d = f(x0) / (fx(x0)**2 + fy(x0)**2 + fz(x0)**2)

        x1[0] = x0[0] - d*fx(x0)
        x1[1] = x0[1] - d*fy(x0)
        x1[2] = x0[2] - d*fz(x0)

        roots.append(x1)

        x0 = x1

    print(tabulate(roots, headers = ["x", "y", "z"]))

    x = []
    y = []
    z = []

    for n in roots:
        x.append(n[0])
        y.append(n[1])
        z.append(n[2])

    [alpha, constant] = convergence(1.09364, x)
    print("Order of convergence of x is", alpha)

    [alpha, constant] = convergence(1.36033, y)
    print("Order of convergence of y is", alpha)

    [alpha, constant] = convergence(1.36033, z)
    print("Order of the convergence of z is", alpha)


def convergence(roots, vector):
    n = len(vector)
    num = abs(vector[(n-1)] - roots)
    denom = abs(vector[(n-2)] - roots)
    if (isinstance(num/denom, float)) and (num/denom < 1):
        alpha = 1
        constant = num/denom
        return [alpha, constant]
    alpha = 2
    while num/(denom**alpha) == float("INF"):
        alpha = alpha + 1
    return [alpha, num/(denom**alpha)]

driver()


