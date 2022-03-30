# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Callable, Union
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import sympy as sym
from scipy.sparse.linalg import norm as spnorm

from matrix_lsq import LeastSquares, Storage, DiskStorage

Matrix = Union[np.ndarray, sp.spmatrix]

"""for nice representation of plots"""

sym.init_printing()
fontsize = 20
new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
              'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize, 'figure.titlesize': fontsize,
              'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
plt.rcParams.update(new_params)


def norm(mat: Matrix) -> float:
    # Frobenius norm of matrices
    # 2-norm of vectors
    if isinstance(mat, sp.spmatrix):
        return spnorm(mat)
    else:
        return np.linalg.norm(mat)


def mls_compute_from_fit(data: np.ndarray, mats: List[Matrix]) -> Matrix:
    if isinstance(mats[0], sp.spmatrix):
        out = sp.csr_matrix(mats[0].shape, dtype=float)
    else:
        out = np.zeros_like(mats[0], dtype=float)
    for i, coeff in enumerate(data.ravel()):
        out += coeff * mats[i]
    return out


class MatrixLSQ:
    storage: Storage
    root: Path

    a1_list: List[Matrix] = None
    a2_list: List[Matrix] = None
    f0_list: List[Matrix] = None
    f1_dir_list: List[Matrix] = None
    f2_dir_list: List[Matrix] = None
    num_kept: int = None

    def __init__(self, root: Path):
        self.root = root
        self.storage = DiskStorage(root)
        self.fitter = LeastSquares(self.storage)

        self.has_non_homo_dirichlet = (self.root / "object-0" / "obj-f1_dir.npy").exists()

    def __call__(self):
        self.a1_list = self.fitter("a1")
        self.a2_list = self.fitter("a2")
        self.f0_list = self.fitter("f0")
        if self.has_non_homo_dirichlet:
            self.f1_dir_list = self.fitter("f1_dir")
            self.f2_dir_list = self.fitter("f2_dir")

        # for now!
        self.num_kept = len(self.a1_list)


if __name__ == '__main__':
    from fem_quadrilateral import ScalableRectangleSolver, DraggableCornerRectangleSolver

    print("dir_bc = 1, 1")


    def dir_bc(x, y):
        return 1, 1


    n = 20
    order = 2
    ant = 5
    root = Path("test_storage_SR1")

    if len(DiskStorage(root)) == 0:
        d = ScalableRectangleSolver(n, 0, dirichlet_bc_func=dir_bc)
        d.matrix_lsq_setup(mls_order=order)
        d.save_snapshots(root, ant)
    d = ScalableRectangleSolver(n, 0, dirichlet_bc_func=dir_bc)
    d.matrix_lsq_setup(mls_order=order)
    print(d.sym_mls_funcs, len(d.sym_mls_funcs))
    print(d.geo_param_range)

    mls = MatrixLSQ(root)
    mls()
    print("-" * 50)

    root = Path("test_storage_DR1")

    if len(DiskStorage(root)) == 0:
        d = DraggableCornerRectangleSolver(n, 0, dirichlet_bc_func=dir_bc)
        d.matrix_lsq_setup(mls_order=order)
        d.save_snapshots(root, ant)
    d = DraggableCornerRectangleSolver(n, 0, dirichlet_bc_func=dir_bc)
    d.matrix_lsq_setup(mls_order=order)
    print(d.sym_mls_funcs, len(d.sym_mls_funcs))
    print(d.geo_param_range)

    mls = MatrixLSQ(root)
    mls()
    print("-" * 50)
