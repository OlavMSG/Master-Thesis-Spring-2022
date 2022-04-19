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


def save_snapshots(p_order):

    n = 20
    mu_grid = 25
    main_root = Path("DR_mls_order_analysis")

    d = DraggableCornerRectangleSolver(n, f_func=f, get_dirichlet_edge_func=clamped_bc)
    d.matrix_lsq_setup(p_order)
    print("p-order:", p_order)
    print("Ant comp:", len(d.sym_mls_funcs))
    print("Ant snapshots:", mu_grid ** 2)
    print("Ratio:", mu_grid ** 2 / (len(d.sym_mls_funcs) * p_order))
    root = main_root / f"p_order_{p_order}"
    print("root:", root)
    print("-" * 50)
    if len(DiskStorage(root)) == 0:
        d.save_snapshots(root, mu_grid)
    else:
        d.matrix_lsq(root)
        print(dict(zip(np.arange(len(d.sym_mls_funcs)), d.sym_mls_funcs)))
    print("-" * 50)


def main():
    print(datetime.now().time())
    max_order = 10
    multiprocess = False
    if multiprocess:
        pool = mp.Pool(mp.cpu_count())
        jobs = []
        for p_order in range(1, max_order + 1):
            job = pool.apply_async(save_snapshots, [p_order])
            jobs.append(job)
        # collect results from the make_plots through the pool result queue
        for job in jobs:
            job.get()
        # now we are done, kill the listener
        pool.close()
        pool.join()
    else:
        for p_order in range(1, max_order + 1):
            save_snapshots(p_order)
    print(datetime.now().time())


if __name__ == '__main__':
    main()
