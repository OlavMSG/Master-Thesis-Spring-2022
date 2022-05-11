# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from time import perf_counter
from fem_quadrilateral import DraggableCornerRectangleSolver, ScalableRectangleSolver, QuadrilateralSolver, \
    default_constants

rho_steal = 8e3  # kg/m^3
alpha = 8e3 * 100 * 9.81 * 0.01  # N/m^2


# Example 2: Gravity in 2D
def f(x, y):
    return alpha, 0


def clamped_bc(x, y):
    return abs(x) <= default_constants.default_tol


def main():
    n = 2
    order = 2
    q = QuadrilateralSolver(n, 0)
    print(q.sym_phi)
    print(q.sym_jac)
    print(q.sym_det_jac)
    q.matrix_lsq_setup(mls_order=order)
    print(q.sym_mls_funcs)
    print(len(q.sym_mls_funcs))
    # q.get_geo_param_limit_estimate(5)
    print("------------")
    d = DraggableCornerRectangleSolver(n, 0)
    print(d.sym_phi)
    print(d.sym_jac)
    print(d.sym_det_jac)
    d.matrix_lsq_setup(mls_order=order)
    print(d.sym_mls_funcs)
    print(len(d.sym_mls_funcs))
    # d.assemble(0.1, 0.2)
    # d.get_geo_param_limit_estimate(101)
    print("------------")
    r = ScalableRectangleSolver(n, 0)
    print(r.sym_phi)
    r.matrix_lsq_setup(mls_order=order)
    print(r.sym_mls_funcs)
    print(len(r.sym_mls_funcs))
    # r.get_geo_param_limit_estimate()

    for n in [20, 40, 80]:
        d = DraggableCornerRectangleSolver(n, f, get_dirichlet_edge_func=clamped_bc)
        s = perf_counter()
        d.assemble(0.2, 0.2)
        print(f"time assemble {n}:", perf_counter() - s)


if __name__ == '__main__':
    main()
