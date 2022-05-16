# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path

import numpy as np
from matrix_lsq import DiskStorage
from src.fem_quadrilateral import DraggableCornerRectangleSolver
from datetime import datetime
import multiprocessing as mp


def save_snapshots(p_order):

    def dir_bc(x, y):
        return x, 0

    n = 20
    mu_grid = 25
    main_root = Path("DR_mls_order_analysis")

    d = DraggableCornerRectangleSolver(n, 0, dirichlet_bc_func=dir_bc)
    d.set_geo_param_range((-0.49, 0.49))
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
        for p_order in range(1, max_order + 1):
            pool.apply_async(save_snapshots, [p_order])
        # now we are done, kill the listener
        pool.close()
        pool.join()
    else:
        for p_order in range(1, max_order + 1):
            save_snapshots(p_order)
    print(datetime.now().time())


if __name__ == '__main__':
    main()
