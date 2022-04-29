# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from pathlib import Path

import numpy as np
from matrix_lsq import DiskStorage

import default_constants
from fem_quadrilateral import DraggableCornerRectangleSolver
from datetime import datetime
import multiprocessing as mp

# rho_steal = 8e3  # kg/m^3
alpha = 8e3 * 100 * 9.81 * 0.01  # N/m^2


# Example 2: Gravity in 2D
def f(x, y):
    return alpha, 0


def clamped_bc(x, y):
    return abs(x) <= default_constants.default_tol


def save_snapshots(n, mu_grid, root, p_order, power_divider=3):
    d = DraggableCornerRectangleSolver(n, f_func=f, get_dirichlet_edge_func=clamped_bc)
    d.matrix_lsq_setup(p_order)
    print("p-order:", p_order)
    print("Ant comp:", len(d.sym_mls_funcs))
    print("Ant snapshots:", mu_grid ** 2)
    print("Ratio:", mu_grid ** 2 / (len(d.sym_mls_funcs) * p_order))
    print("root:", root)
    print("-" * 50)
    print(dict(zip(np.arange(len(d.sym_mls_funcs)), d.sym_mls_funcs)))
    d.multiprocessing_save_snapshots(root, mu_grid, power_divider=power_divider)
    print("-" * 50)


def main():
    max_order = 25
    power_divider = 2
    n = 20
    mu_grid = 25
    main_root = Path("DR_mls_order_analysis")
    print(datetime.now().time())
    for p_order in range(1, max_order + 1):
        root = main_root / f"p_order_{p_order}"
        if len(DiskStorage(root)) != mu_grid ** 2:
            print(datetime.now().time())
            save_snapshots(n, mu_grid, root, p_order, power_divider)
            print(datetime.now().time())
    print(datetime.now().time())


if __name__ == '__main__':
    main()
