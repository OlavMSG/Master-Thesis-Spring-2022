# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import numpy as np
import sympy as sym
from sympy.abc import x, y, a, b, s, t
from src.fem_quadrilateral import QuadrilateralSolver

from scipy.special import legendre


def leg(x, n):
    Leg = legendre(n)
    y = Leg(x)
    return y


def main():
    sym_f = x ** 2 + y ** 2
    sym_f_grad = [sym_f.diff(z) for z in [x, y]]
    print(sym_f)
    print(sym_f_grad)

    f = sym.lambdify([x, y], sym_f, "numpy")
    print(type(f))
    print(f(0, 0))
    f2 = lambda x, y: x ** 2 + y ** 2
    print(type(f2))
    print(f2(0, 0))

    f = 1 / (1 + a * x + b * y)
    f2 = f.subs({a: a * s, b: b * s})
    g = sym.series(f2, s, 0, 3, "-").removeO().subs({s: 1})

    print(f)
    print(f2)
    print(g)

    z = [x, y]
    z2 = sym.lambdify([x, y], z, "numpy")

    x1 = np.arange(2) + 1
    y1 = np.arange(2) + 3
    print(x1, y1)
    z3 = z2(*x1)
    print(z3)
    print(type(z3))

    z4 = lambda x, y: np.array([[x, y],
                                [x * y, x / y]])
    z5 = z4(x1, y1)

    print(z5)

def compute_num_QS_sanpshots(n, p):
    num = n ** 6
    q = QuadrilateralSolver(2, 0)
    q.matrix_lsq_setup(p)
    ant = len(q.sym_mls_funcs)
    print("ant snapshots:", num)
    print("ant real snapshots:", num * 5 ** 2)
    print("ant mls funcs:", ant)
    print("ratio:", ant / num)
    print(ant / num < 0.30)
    step = (q.geo_param_range[1] - q.geo_param_range[0]) / num
    step_goal = (0.5 + 0.5) / 25 ** 2
    print("step:", step)
    print("step_goal:", step_goal)
    print("-"*50)
    # print(q.sym_mls_funcs)
    print("_"*50)


if __name__ == '__main__':
    # main()
    compute_num_QS_sanpshots(7, 9)


