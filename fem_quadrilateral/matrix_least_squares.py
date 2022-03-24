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
from fem_quadrilateral.base_solver import BaseSolver

from matrix_lsq import LeastSquares, Storage, DiskStorage, Snapshot

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

    def analyze(self, solver: BaseSolver):
        # get mean matrix
        root_mean = self.root / "mean"
        mean_snapshot = Snapshot(root_mean)
        arg = np.argwhere(mean_snapshot.data == 0)
        data = mean_snapshot.data
        data[arg] = 1e-8
        # compute matrices and norms
        a1_norm = norm(mls_compute_from_fit(data, self.a1_list))
        a2_norm = norm(mls_compute_from_fit(data, self.a2_list))
        f0_norm = norm(mls_compute_from_fit(data, self.f0_list))

        a1_norm_list = np.array(list(map(norm, self.a1_list)))
        a2_norm_list = np.array(list(map(norm, self.a2_list)))
        f0_norm_list = np.array(list(map(norm, self.f0_list)))

        if self.has_non_homo_dirichlet:
            # compute matrices and norms
            f1_dir_norm = norm(mls_compute_from_fit(data, self.f1_dir_list))
            f2_dir_norm = norm(mls_compute_from_fit(data, self.f2_dir_list))
            # compute norms
            f1_dir_norm_list = np.array(list(map(norm, self.f1_dir_list)))
            f2_dir_norm_list = np.array(list(map(norm, self.f2_dir_list)))

        print("a1 norm f*norm rel_norm a1_norm")
        print(a1_norm_list)
        print(np.abs(data.ravel()) * a1_norm_list)
        print(np.abs(data.ravel()) * a1_norm_list / a1_norm)
        print(a1_norm)
        print("a2 norm f*norm rel_norm a2_norm")
        print(a2_norm_list)
        print(np.abs(data.ravel()) * a2_norm_list)
        print(np.abs(data.ravel()) * a2_norm_list / a2_norm)
        print(a2_norm)
        print("f0 norm f*norm (rel_norm) f0_norm")
        print(f0_norm_list)
        print(np.abs(data.ravel()) * f0_norm_list)
        if f0_norm != 0:
            print(np.abs(data.ravel()) * f0_norm_list / f0_norm)
        print(f0_norm)
        if self.has_non_homo_dirichlet:
            print("f1_dir norm f*norm rel_norm f1_dir_norm")
            print(f1_dir_norm_list)
            print(np.abs(data.ravel()) * f1_dir_norm_list)
            print(np.abs(data.ravel()) * f1_dir_norm_list / f1_dir_norm)
            print(f1_dir_norm)
            print("f2_dir norm f*norm rel_norm f2_dir_norm")
            print(f2_dir_norm_list)
            print(np.abs(data.ravel()) * f2_dir_norm_list)
            print(np.abs(data.ravel()) * f2_dir_norm_list / f2_dir_norm)
            print(f2_dir_norm)

        save_dict = "".join(("plots_", str(self.root)))
        Path(save_dict).mkdir(parents=True, exist_ok=True)

        plt.figure("a-1")
        plt.title("a - norm")
        plt.semilogy(a1_norm_list, "x--", label="$norm(A_{1q})$")
        plt.semilogy(np.abs(data.ravel()) * a1_norm_list, "x--", label="$norm(f_q(\\mu_{mean})A_{1q})$")
        plt.semilogy(a2_norm_list, "x--", label="$norm(A_{2q})$")
        plt.semilogy(np.abs(data.ravel()) * a2_norm_list, "x--", label="$norm(f_q(\\mu_{mean})A_{2q})$")
        plt.xticks(np.arange(len(solver.sym_mls_funcs)), map(str, solver.sym_mls_funcs), rotation=90)
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.43), ncol=2)
        plt.grid()
        plt.savefig("".join((save_dict, "\\a-norm.pdf")), bbox_inches='tight')

        plt.figure("a-2")
        plt.title("a - relative norm")
        plt.semilogy(np.abs(data.ravel()) * a1_norm_list / a1_norm, "x--",
                     label="$norm(f_q(\\mu_{mean})A_{2q}) / norm(A(\\mu_{mean}))$")
        plt.semilogy(np.abs(data.ravel()) * a2_norm_list / a2_norm, "x--",
                     label="$norm(f_q(\\mu_{mean})A_{2q}) / norm(A(\\mu_{mean}))$")
        plt.xticks(np.arange(len(solver.sym_mls_funcs)), map(str, solver.sym_mls_funcs), rotation=90)
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.43), ncol=2)
        plt.grid()
        plt.savefig("".join((save_dict, "\\a-relative_norm.pdf")), bbox_inches='tight')

        plt.figure("f-1")
        plt.title("f - norm")
        plt.semilogy(f0_norm_list, "x--", label="$norm(f0_{q})$")
        plt.semilogy(np.abs(data.ravel()) * f0_norm_list, "x--", label="$norm(f_q(\\mu_{mean})f0_{q})$")
        if self.has_non_homo_dirichlet:
            plt.semilogy(f1_dir_norm_list, "x--", label="$norm(f1_{q})$")
            plt.semilogy(data.ravel() * f1_dir_norm_list, "x--", label="$norm(f_q(\\mu_{mean})f1_{q})$")
            plt.semilogy(f2_dir_norm_list, "x--", label="$norm(f2_{q})$")
            plt.semilogy(data.ravel() * f2_dir_norm_list, "x--", label="$norm(f_q(\\mu_{mean})f2_{q})$")
        plt.xticks(np.arange(len(solver.sym_mls_funcs)), map(str, solver.sym_mls_funcs), rotation=90)
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.43), ncol=2)
        plt.grid()
        plt.savefig("".join((save_dict, "\\f-norm.pdf")), bbox_inches='tight')

        plt.figure("f-2")
        plt.title("f - relative norm")
        if f0_norm != 0:
            plt.semilogy(np.abs(data.ravel()) * f0_norm_list / f0_norm, "x--",
                         label="$norm(f_q(\\mu_{mean})f0_{q}) / norm(f0(\\mu_{mean}))$")
        if self.has_non_homo_dirichlet:
            plt.semilogy(np.abs(data.ravel()) * f1_dir_norm_list / f1_dir_norm, "x--",
                         label="$norm(f_q(\\mu_{mean})f1_{q}) / norm(f1(\\mu_{mean}))$")
            plt.semilogy(np.abs(data.ravel()) * f2_dir_norm_list / f2_dir_norm, "x--",
                         label="$norm(f_q(\\mu_{mean})f2_{q}) / norm(f2(\\mu_{mean}))$")
        plt.xticks(np.arange(len(solver.sym_mls_funcs)), map(str, solver.sym_mls_funcs), rotation=90)
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.43), ncol=2)
        plt.grid()
        plt.savefig("".join((save_dict, "\\f-relative_norm.pdf")), bbox_inches='tight')

        plt.show()


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
    mls.analyze(d)
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
    mls.analyze(d)
    print("-" * 50)

    print("dir_bc = x, 0")


    def dir_bc(x, y):
        return x, 0


    root = Path("test_storage_SR2")

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
    mls.analyze(d)
    print("-" * 50)

    root = Path("test_storage_DR2")

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
    mls.analyze(d)
    print("-" * 50)
