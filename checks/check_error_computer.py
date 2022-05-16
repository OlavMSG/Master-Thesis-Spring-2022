# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from pathlib import Path

import numpy as np

from src.fem_quadrilateral import default_constants


def main():
    from src.fem_quadrilateral import ScalableRectangleSolver
    from src.fem_quadrilateral import RbErrorComputer, HfErrorComputer
    from matrix_lsq import DiskStorage

    def dir_bc(x, y):
        return x, 0

    n = 20
    root = Path("../test_storage1")
    if len(DiskStorage(root)) == 0:
        r = ScalableRectangleSolver(n, 0, dirichlet_bc_func=dir_bc, x0=1.1, y0=2.1)
        r.matrix_lsq_setup(2)
        r.save_snapshots(root, 5)

    r = ScalableRectangleSolver.from_root(root)
    r.build_rb_model(root)
    r.matrix_lsq_setup()

    r2 = ScalableRectangleSolver(n, 0, dirichlet_bc_func=dir_bc, x0=1.1, y0=2.1)
    r2.build_rb_model(root)
    r2.matrix_lsq_setup()

    err_rb = RbErrorComputer(root)
    err_hf = HfErrorComputer(root)

    e_mean = np.mean(default_constants.e_young_range)
    nu_mean = np.mean(default_constants.nu_poisson_range)
    geo_mean = 3.85  # np.mean(r.geo_param_range

    print(r.mls_order)
    print("RB error")
    err_rb(r, e_mean, nu_mean, geo_mean, geo_mean)
    print("-" * 50)
    err_rb(r2, e_mean + 1, nu_mean, geo_mean, geo_mean)
    print("-" * 50)
    err_rb(r2, e_mean, nu_mean, geo_mean + 1, geo_mean)
    # see if error
    # err_rb(r, e_mean+1, nu_mean, geo_mean, geo_mean)
    print("-" * 50)
    print("-" * 50)
    print("HF error")
    err_hf(r, e_mean, nu_mean, geo_mean, geo_mean)
    print("-" * 50)
    err_hf(r2, e_mean + 1, nu_mean, geo_mean, geo_mean)
    print("-" * 50)
    err_hf(r2, e_mean, nu_mean, geo_mean + 1, geo_mean)
    # see if error
    # err_hf(r, e_mean+1, nu_mean, geo_mean, geo_mean)
    print("-" * 50)
    print("-" * 50)


if __name__ == '__main__':
    main()
