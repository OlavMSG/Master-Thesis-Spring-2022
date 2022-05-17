# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import numpy as np

from src.fem_quadrilateral import DraggableCornerRectangleSolver
from src.fem_quadrilateral.default_constants import e_young_range, nu_poisson_range


def base(n, tol, dirichlet_bc_func, u_exact_func, f=None, element="lt", nq=None):
    if f is None:
        f = 0
    e_mean = np.mean(e_young_range)
    nu_mean = np.mean(nu_poisson_range)
    for bcs_bool in [True, False]:
        s_rec = DraggableCornerRectangleSolver(n, f, dirichlet_bc_func=dirichlet_bc_func, element=element,
                                               bcs_are_on_reference_domain=bcs_bool)
        if nq is not None:
            s_rec.set_quadrature_scheme_order(nq)
        s_rec.assemble(mu1, mu2)

        s_rec.hfsolve(e_mean, nu_mean, print_info=False)
        u_exact = s_rec.get_u_exact(u_exact_func)

        # discrete max norm, holds if u_exact is triangle (Terms 1, x, y)
        test_res = np.all(np.abs(s_rec.uh_full - u_exact.flatt_values) < tol)
        print(f"BC are on ref: {bcs_bool}")
        print("max norm {}".format(np.max(np.abs(s_rec.uh_full - u_exact.flatt_values))))
        print("quadrature rule {}x{}".format(s_rec.nq, s_rec.nq_y))
        print("tolerance {}".format(tol))
        print("ref plate limits {}".format(s_rec.ref_plate))
        print("element type: {}".format(element))
        print("test {} for n={}".format(test_res, n))

        print("-" * 10)


def case_1(n, tol, element="lt", nq=None):
    print("Case 1: (x, 0)")

    def u_exact_func(x, y):
        return x, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(n, tol, dirichlet_bc_func, u_exact_func, element=element, nq=nq)


def case_2(n, tol, element="lt", nq=None):
    print("Case 2: (0, y)")

    def u_exact_func(x, y):
        return 0., y

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(n, tol, dirichlet_bc_func, u_exact_func, element=element, nq=nq)


def case_3(n, tol, element="lt", nq=None):
    print("Case 3: (y, 0)")

    def u_exact_func(x, y):
        return y, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(n, tol, dirichlet_bc_func, u_exact_func, element=element, nq=nq)


def case_4(n, tol, element="lt", nq=None):
    print("Case 4: (0, x)")

    def u_exact_func(x, y):
        return 0., x

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(n, tol, dirichlet_bc_func, u_exact_func, element=element, nq=nq)


def main(run_for=False):
    n = 2
    tol = 1e-14

    case_1(n, tol, element="bq")
    case_2(n, tol, element="bq")
    case_3(n, tol, element="bq")
    case_4(n, tol, element="bq")

    print("*" * 40)
    if run_for:
        for nq in (3, 4, 6):
            case_1(n, tol, element="lt", nq=nq)
        for nq in (3, 4, 6):
            case_2(n, tol, element="lt", nq=nq)
        for nq in (3, 4, 6):
            case_3(n, tol, element="lt", nq=nq)
        for nq in (3, 4, 6):
            case_4(n, tol, element="lt", nq=nq)
    else:
        nq = 4
        case_1(n, tol, element="lt", nq=nq)
        case_2(n, tol, element="lt", nq=nq)
        case_3(n, tol, element="lt", nq=nq)
        case_4(n, tol, element="lt", nq=nq)


if __name__ == '__main__':
    DraggableCornerRectangleSolver.mu_to_vertices_dict()
    # play with the numbers'
    # element="lt" passes for some
    # element="bq" passes for all tested so far...
    mu1 = 0.2
    mu2 = 0.2
    main(False)
