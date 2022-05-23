# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from itertools import product, repeat

import numpy as np
from tqdm import tqdm

from fem_quadrilateral import QuadrilateralSolver
from fem_quadrilateral.default_constants import e_young_range, nu_poisson_range


def base(n, tol, dirichlet_bc_func, u_exact_func, mus, f=None, element="br", ifprint=False):
    if f is None:
        f = 0
    e_mean = np.mean(e_young_range)
    nu_mean = np.mean(nu_poisson_range)
    s_rec = QuadrilateralSolver(n, f, dirichlet_bc_func=dirichlet_bc_func, element=element,
                                bcs_are_on_reference_domain=False)
    s_rec.set_geo_param_range((-0.16, 0.16))
    s_rec.assemble(*mus)

    s_rec.hfsolve(e_mean, nu_mean, print_info=False)
    u_exact = s_rec.get_u_exact(u_exact_func)

    # discrete max norm, holds if u_exact is triangle (Terms 1, x, y)
    test_res = np.all(np.abs(s_rec.uh_full - u_exact.flatt_values) < tol)
    if ifprint:
        print("max norm {}".format(np.max(np.abs(s_rec.uh_full - u_exact.flatt_values))))
        print("tolerance {}".format(tol))
        print("plate limits {}".format(s_rec.ref_plate))
        print("element type: {}".format(element))
        print("test {} for n={}".format(test_res, n))
        print("free_node_ref:", s_rec.p[s_rec.free_index, :])
        print("free_node:", s_rec.vectorized_phi(s_rec.p[s_rec.free_index, 0], s_rec.p[s_rec.free_index, 1], *mus))
        print("u_ex[free_node]:", u_exact.flatt_values[s_rec.expanded_free_index])
        print("u_h[free_node]:", s_rec.uh_free)
        print("-" * 10)
    else:
        assert test_res


def case_1(n, tol, mus, element="br", ifprint=False):
    if ifprint:
        print("Case 1: (x, 0)")

    def u_exact_func(x, y):
        return x, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(n, tol, dirichlet_bc_func, u_exact_func, mus, element=element, ifprint=ifprint)


def case_2(n, tol, mus, element="br", ifprint=False):
    if ifprint:
        print("Case 2: (0, y)")

    def u_exact_func(x, y):
        return 0., y

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(n, tol, dirichlet_bc_func, u_exact_func, mus, element=element, ifprint=ifprint)


def case_3(n, tol, mus, element="br", ifprint=False):
    if ifprint:
        print("Case 3: (y, 0)")

    def u_exact_func(x, y):
        return y, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(n, tol, dirichlet_bc_func, u_exact_func, mus, element=element, ifprint=ifprint)


def case_4(n, tol, mus, element="br", ifprint=False):
    if ifprint:
        print("Case 4: (0, x)")

    def u_exact_func(x, y):
        return 0., x

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(n, tol, dirichlet_bc_func, u_exact_func, mus, element=element, ifprint=ifprint)


def run_patch_test(m=2):
    n = 2
    tol = 1e-14
    l_list = np.linspace(-0.16, 0.16, m + 1)
    for mus in tqdm(product(*repeat(l_list, 6)), desc="Testing"):
        case_1(n, tol, mus, element="br")
        case_2(n, tol, mus, element="br")
        case_3(n, tol, mus, element="br")
        case_4(n, tol, mus, element="br")
    print("All True")


def main(mus):
    n = 2
    tol = 1e-14
    QuadrilateralSolver.mu_to_vertices_dict()
    # case_1(n, tol, element="lt")
    # case_2(n, tol, element="lt")
    # case_3(n, tol, element="lt")
    # case_4(n, tol, element="lt")
    # print("*"*40)
    case_1(n, tol, mus, element="br", ifprint=True)
    case_2(n, tol, mus, element="br", ifprint=True)
    case_3(n, tol, mus, element="br", ifprint=True)
    case_4(n, tol, mus, element="br", ifprint=True)


if __name__ == '__main__':
    # run_patch_test(2)
    mu1, mu2, mu3, mu4, mu5, mu6 = -0.1, -0.1, -0.1, 0.1, 0.1, 0.1
    mus = [mu1, mu2, mu3, mu4, mu5, mu6]
    main(mus)
