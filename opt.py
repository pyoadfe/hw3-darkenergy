#!/usr/bin/env python3

from collections import namedtuple
import numpy as np

Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов модельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива — менее nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""

def gauss_newton(y, f, j, x0, k=1, tol=1e-4):
    x = np.asarray(x0, dtype=float)
    nfev = 0
    cost = []
    dx = 1.0                                                    #First step
    J = 0                                                       #
    r = 0                                                       #To be able to read after loop
    while np.linalg.norm(dx) >= tol:
        r = y - f(*x)                                           #Residial vector
        J = j(*x)                                               #Jacobian of f
        dx = np.linalg.inv(J.T @ J) @ J.T @ r                   #Step of Algorithm
        x += dx
        nfev += 1
        cost.append(0.5 * r @ r)
    
    return Result(nfev = nfev,
                  cost = np.asarray(cost, dtype=float),
                  gradnorm = np.linalg.norm(J.T @ r),
                  x = x)

def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4):
    x = np.asarray(x0, dtype=float)
    nfev = 0
    cost = []
    dx = 1
    G = 0
    while np.linalg.norm(dx) >= tol:
        r = y - f(*x)
        J = j(*x)
        G = J.T @ r
        F = 0.5 * r @ r

        dx1 = np.linalg.inv(J.T @ J + lmbd0) @ G
        dx2 = np.linalg.inv(J.T @ J + lmbd0 / nu) @ G
        r1 = y - f(*(x + dx1))
        r2 = y - f(*(x + dx2))
        F1 = 0.5 * r1 @ r1
        F2 = 0.5 * r2 @ r2

        if F2 <= F:
            cost.append(F)
            lmbd0 /= nu 
            dx = dx2 
        elif F1 <= F:
            dx = dx1
        else:
            cnt = 0
            while F1 > F:
                cnt += 1 
                lmbd0 *= nu**i
                dx = np.linalg.inv(J.T @ J + lmbd0) @ G
                r_tmp = y - f(*(x + dx))
                F1 = 0.5 * r_tmp @ r_tmp
            cost.append(F1)
            lmbd0 *= nu**i

        x += dx
        nfev += 1

    return Result(nfev = nfev,
                  cost = np.asarray(cost, dtype=float),
                  gradnorm = np.linalg.norm(G),
                  x = x)
