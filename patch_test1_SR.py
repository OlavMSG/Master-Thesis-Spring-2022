# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import sys

import numpy as np

from fem_quadrilateral_classes import ScalableRectangleSolver
from default_constants import e_young_range, nu_poisson_range
from helpers import get_u_exact


def base(n, tol, dirichlet_bc_func, u_exact_func, f=None, element="lt"):
    if f is None:
        f = 0
    e_mean = np.mean(e_young_range)
    nu_mean = np.mean(nu_poisson_range)

    le2d = ScalableRectangleSolver(n, f, dirichlet_bc_func=dirichlet_bc_func, element=element)
    le2d.assemble(lx, ly)

    le2d.hfsolve(e_mean, nu_mean, print_info=False)
    u_exact = get_u_exact(le2d.p, u_exact_func)

    # discrete max norm, holds if u_exact is linear (Terms 1, x, y)
    test_res = np.all(np.abs(le2d.uh_full - u_exact.flatt_values) < tol)
    print("max norm {}".format(np.max(np.abs(le2d.uh_full - u_exact.flatt_values))))
    print("tolerance {}".format(tol))
    print("plate limits {}".format(le2d.ref_plate))
    print("element type: {}".format(element))
    print("test {} for n={}".format(test_res, n))

    print("-" * 10)

    assert test_res


def case_1(n, tol, element="lt"):
    print("Case 1: (x, 0)")

    def u_exact_func(x, y):
        return x, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(n, tol, dirichlet_bc_func, u_exact_func, element=element)


def case_2(n, tol, element="lt"):
    print("Case 2: (0, y)")

    def u_exact_func(x, y):
        return 0., y

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(n, tol, dirichlet_bc_func, u_exact_func, element=element)


def case_3(n, tol, element="lt"):
    print("Case 3: (y, 0)")

    def u_exact_func(x, y):
        return y, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(n, tol, dirichlet_bc_func, u_exact_func, element=element)


def case_4(n, tol, element="lt"):
    print("Case 4: (0, x)")

    def u_exact_func(x, y):
        return 0., x

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(n, tol, dirichlet_bc_func, u_exact_func, element=element)


def main():
    n = 2
    tol = 1e-14
    case_1(n, tol, element="lt")
    case_2(n, tol, element="lt")
    case_3(n, tol, element="lt")
    case_4(n, tol, element="lt")

    case_1(n, tol, element="bq")
    case_2(n, tol, element="bq")
    case_3(n, tol, element="bq")
    case_4(n, tol, element="bq")


if __name__ == '__main__':
    lx = 2
    ly = 3
    main()

