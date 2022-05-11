# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from pathlib import Path

import numpy as np
from matrix_lsq import DiskStorage

from fem_quadrilateral import QuadrilateralSolver, default_constants
from datetime import datetime

# rho_steal = 8e3  # kg/m^3
alpha = 8e3 * 100 * 9.81 * 0.01  # N/m^2


# Example 2: Gravity in 2D
def f(x, y):
    return alpha, 0


def clamped_bc(x, y):
    return abs(x) <= default_constants.default_tol


def save_snapshots(p_order, power_divider):
    n = 20
    mu_grid = 5  # gives 15_625 snapshots...
    main_root = Path("QS_mls_order_analysis")

    d = QuadrilateralSolver(n, f_func=f, get_dirichlet_edge_func=clamped_bc)
    d.matrix_lsq_setup(p_order)
    print("p-order:", p_order)
    print("Ant comp:", len(d.sym_mls_funcs))
    print("Ant snapshots:", mu_grid ** 6)
    print("Ratio:", mu_grid ** 6 / (len(d.sym_mls_funcs) * p_order))
    root = main_root / f"p_order_{p_order}"
    print("root:", root)
    print("-" * 50)
    print(dict(zip(np.arange(len(d.sym_mls_funcs)), d.sym_mls_funcs)))
    check_running_folder = main_root / "check_running_folder"
    check_running_folder.mkdir(parents=True, exist_ok=True)
    if len(DiskStorage(root)) != mu_grid ** 6:
        d.multiprocessing_save_snapshots(root, mu_grid, power_divider=power_divider)
        # use to stopp making more if this fails
        try:
            q = QuadrilateralSolver.from_root(root)
            q.matrix_lsq_setup()
            q.matrix_lsq(root)
            print("-" * 50)
        except Exception as e:
            # delete running folder
            check_running_folder.unlink()
            raise e
    else:
        q = QuadrilateralSolver.from_root(root)
        q.matrix_lsq_setup()
        q.matrix_lsq(root)
        print("-" * 50)


def main():
    print(datetime.now().time())
    max_order = 10
    power_divider = 3
    for p_order in range(10, max_order + 1):
        save_snapshots(p_order, power_divider)
    print(datetime.now().time())


if __name__ == '__main__':
    main()
