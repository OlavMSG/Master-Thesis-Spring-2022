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
    """d = GetSolver("DR")
    d = d(2, f, get_dirichlet_edge_func=clamped_bc, bcs_are_on_reference_domain=True)
    d.assemble(0.2, 0.2)

    d.hfsolve(210e3, 0.2)
    print(d.f0)
    d = GetSolver("DR")
    d = d(2, f, get_dirichlet_edge_func=clamped_bc, bcs_are_on_reference_domain=False)
    d.assemble(0.2, 0.2)

    d.hfsolve(210e3, 0.2)
    print(d.f0)"""
    """print(d.det_jac_func(-1, -1, -1, -1))
    d.plot_mesh(0.6, 0.2)
    plt.show()
    
    
    d.hf_plot_displacement()
    plt.show()
    d.hf_von_mises_stress()
    d.hf_plot_von_mises()
    plt.show()
    # d.rb_pod_mode(1)

    print(d.jac_phi_inv(0.5, 0.5, 0.5, -0.2))
    print(0 < 1 < np.inf)
    print(d.tri)"""

    from matrix_lsq import Snapshot, DiskStorage
    from pathlib import Path
    root = Path(r"C:\Users\Xilva98\Documents\Masteroppgave - Vår 2022\Master-Thesis-Spring-2022\DR_mls_analysis_new\DR_mls_order_analysis\p_order_1")
    """storage = DiskStorage(root)
    geo_vec = helpers.get_vec_from_range(DraggableCornerRectangleSolver.geo_param_range, 25, "uniform")
    geo_mat = np.array(list(product(geo_vec, repeat=2)))
    for i, snapshot in enumerate(storage):
        d = DraggableCornerRectangleSolver(20, f, get_dirichlet_edge_func=clamped_bc, bcs_are_on_reference_domain=False)
        d.assemble(*geo_mat[i, :])
        # print(d.rg)
        # root_mean = root / "mean"
        # rg = Snapshot(root_mean)["rg"]
        # print(rg)
        # print(np.all(np.abs(d.rg - rg) < 1e-14))
        print(d.f0)
        print(snapshot["f0"])
        print(np.all(np.abs(d.f0 - snapshot["f0"]) < 1e-14))"""

    """d = DraggableCornerRectangleSolver(20, f, get_dirichlet_edge_func=clamped_bc, bcs_are_on_reference_domain=False)
    d.set_geo_param_range((-0.49, 0.49))
    d.assemble(-0.49, -0.49)"""

if __name__ == '__main__':
    main()
