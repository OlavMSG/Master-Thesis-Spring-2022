# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib import pyplot as plt

from fem_quadrilateral import DraggableCornerRectangleSolver, ScalableRectangleSolver, QuadrilateralSolver, \
    default_constants, GetSolver, helpers

rho_steal = 8e3  # kg/m^3
alpha = 8e3 * 100 * 9.81 * 0.01  # N/m^2


# Example 2: Gravity in 2D
def f(x, y):
    return alpha, 0


def clamped_bc(x, y):
    return abs(x) <= default_constants.default_tol


def main():
    n = 2
    order = 1
    """q = QuadrilateralSolver(n, 0)
    print(q.sym_phi)
    print(q.sym_jac)
    print(q.sym_det_jac)
    q.matrix_lsq_setup(mls_order=order)
    print(q.sym_mls_funcs)
    print(len(q.sym_mls_funcs))"""
    # q.get_geo_param_limit_estimate(5)
    """print("------------")
    d = DraggableCornerRectangleSolver(n, 0)
    print(d.sym_phi)
    print(d.sym_jac)
    print(d.sym_det_jac)
    d.matrix_lsq_setup(mls_order=order)
    print(d.sym_mls_funcs)
    print(len(d.sym_mls_funcs))"""
    # d.assemble(0.1, 0.2)
    # d.get_geo_param_limit_estimate(101)
    """print("------------")
    r = ScalableRectangleSolver(n, 0)
    print(r.sym_phi)
    r.matrix_lsq_setup(mls_order=order)
    print(r.sym_mls_funcs)
    print(len(r.sym_mls_funcs))"""
    # r.get_geo_param_limit_estimate()

    """for n in [20, 40, 80]:
        d = DraggableCornerRectangleSolver(n, f, get_dirichlet_edge_func=clamped_bc)
        s = perf_counter()
        d.assemble(0.2, 0.2)
        print(f"time assemble {n}:", perf_counter() - s)"""
    r = ScalableRectangleSolver(4, f, get_dirichlet_edge_func=clamped_bc, bcs_are_on_reference_domain=False)
    r.plot_mesh(0.1, 5.1)
    plt.savefig("plots/test_mesh_SR1.pdf", bbox_inches='tight')
    plt.show()
    r.plot_mesh(5.1, 0.1)
    plt.savefig("plots/test_mesh_SR2.pdf", bbox_inches='tight')
    plt.show()

    r.assemble(5.1, 0.1)
    r.hfsolve(210e3, 0.2)
    r.hf_plot_displacement()
    plt.savefig("plots/test_hf_dis_SR.pdf", bbox_inches='tight')
    plt.show()
    r.hf_von_mises_stress()
    r.hf_plot_von_mises()
    plt.savefig("plots/test_hf_miss_SR.pdf", bbox_inches='tight')
    plt.show()

    d = GetSolver("DR")
    d = d(2, f, get_dirichlet_edge_func=clamped_bc, bcs_are_on_reference_domain=False)
    d.assemble(0.3, 0.3)

    d.hfsolve(210e3, 0.2)
    print(d.det_jac_func(-1, -1, -1, -1))
    d.plot_mesh(0.3, 0.3)
    plt.savefig("plots/test_mesh_DR.pdf", bbox_inches='tight')
    plt.show()

    d.hf_plot_displacement()
    plt.savefig("plots/test_hf_dis_DR.pdf", bbox_inches='tight')
    plt.show()
    d.hf_von_mises_stress()
    d.hf_plot_von_mises()
    plt.savefig("plots/test_hf_miss_DR.pdf", bbox_inches='tight')
    plt.show()
    # d.rb_pod_mode(1)

    print(d.jac_phi_inv(0.5, 0.5, 0.5, -0.2))
    print(0 < 1 < np.inf)
    print(d.tri)


if __name__ == '__main__':
    main()
