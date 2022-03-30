# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

import scipy.sparse as sp
import tqdm
from scipy.linalg import eigh, fractional_matrix_power

import default_constants
from matrix_lsq import Storage, DiskStorage, Snapshot
import numpy as np

Matrix = Union[np.ndarray, sp.spmatrix]


class PodWithEnergyNorm:
    root: Path
    storage: Storage
    eps_pod: float = default_constants.eps_pod
    n_rom_max: int = None
    n_rom: int = None
    sigma2_vec: np.ndarray = None
    i_n: np.ndarray = None
    v_mat_n_max: np.ndarray = None
    v: np.ndarray = None

    def __init__(self, root: Path, eps_pod: Optional[float] = None):
        self.root = root
        self.storage = DiskStorage(root)
        if eps_pod is not None:
            self.eps_pod = eps_pod

        assert len(self.storage) != 0

    def __call__(self):
        # get mean matrix
        root_mean = self.root / "mean"
        mean_snapshot = Snapshot(root_mean)
        a_mean = mean_snapshot["a"]
        n_free = a_mean.shape[0]
        geo_gird, material_grid, num_geo_param = mean_snapshot["grid_params"]
        m = material_grid ** 2
        ns = geo_gird ** num_geo_param * m

        s_mat = np.zeros((n_free, ns), dtype=float)

        for i, snapshot in tqdm.tqdm(enumerate(self.storage), desc="Loading data"):
            s_mat[:, i * m:(i + 1) * m] = snapshot["s_mat"]
        if (s_mat == 0).all():
            error_text = "Solution matrix is zero, can not compute POD for building a reduced model. " \
                         + "The most likely cause is f_func=0, dirichlet_bc_func=0 and neumann_bc_func=0, " \
                         + "where two last may be None."
            raise ValueError(error_text)

        if ns <= n_free:
            # build correlation matrix
            corr_mat = s_mat.T @ a_mean @ s_mat
            # find the eigenvalues and eigenvectors of it
            self.sigma2_vec, zeta_mat = eigh(corr_mat)
            # reverse arrays because they are in ascending order
            self.sigma2_vec = self.sigma2_vec[::-1]
            zeta_mat = zeta_mat[:, ::-1]
            # compute n_rom from relative information content
            self.i_n = np.cumsum(self.sigma2_vec) / np.sum(self.sigma2_vec)
            self.n_rom = np.min(np.argwhere(self.i_n >= 1 - self.eps_pod ** 2)) + 1

            # self.n_rom_max = np.min(np.argwhere(self.sigma2_vec < 0)) + 1
            self.n_rom_max = np.linalg.matrix_rank(s_mat)
            self.v_mat_n_max = s_mat @ zeta_mat[:, :self.n_rom_max] / np.sqrt(self.sigma2_vec[:self.n_rom_max])

        else:
            x05 = fractional_matrix_power(a_mean.A, 0.5)
            # build correlation matrix
            corr_mat = x05 @ s_mat @ s_mat.T @ x05
            # find the eigenvalues and eigenvectors of it
            self.sigma2_vec, zeta_mat = eigh(corr_mat)
            # reverse arrays because they are in ascending order
            self.sigma2_vec = self.sigma2_vec[::-1]
            zeta_mat = zeta_mat[:, ::-1]
            # compute n_rom from relative information content
            self.i_n = np.cumsum(self.sigma2_vec) / np.sum(self.sigma2_vec)
            self.n_rom = np.min(np.argwhere(self.i_n >= 1 - self.eps_pod ** 2)) + 1

            # self.n_rom_max = np.min(np.argwhere(self.sigma2_vec < 0)) + 1
            self.n_rom_max = np.linalg.matrix_rank(s_mat)
            self.v_mat_n_max = np.linalg.solve(x05, zeta_mat[:, :self.n_rom_max])

        self.v = self.v_mat_n_max[:, :self.n_rom]

    def get_v_mat(self, n_rom: int) -> np.ndarray:
        if n_rom > self.n_rom_max:
            raise ValueError(f"No V matrix available for n_rom={n_rom}, max n_rom is {self.n_rom_max}.")
        else:
            return self.v_mat_n_max[:, :n_rom]

    def compute_rom(self, obj: Matrix, n_rom: Optional[int] = None) -> Matrix:
        if n_rom is None:
            if obj.ndim == 1:
                # vector
                return self.v.T @ obj
            elif obj.ndim == 2:
                # matrix
                return self.v.T @ obj @ self.v
        else:
            v = self.get_v_mat(n_rom)
            if obj.ndim == 1:
                # vector
                return v.T @ obj
            elif obj.ndim == 2:
                # matrix
                return v.T @ obj @ v


if __name__ == '__main__':
    from fem_quadrilateral import DraggableCornerRectangleSolver


    def dir_bc(x, y):
        return x, 0


    n = 2
    root = Path("test_storage2")

    if len(DiskStorage(root)) == 0:
        d = DraggableCornerRectangleSolver(3, 0, dirichlet_bc_func=dir_bc)
        d.save_snapshots(root, 3)
    d = DraggableCornerRectangleSolver(3, 0, dirichlet_bc_func=dir_bc)
    d.assemble(0, 0)

    p = PodWithEnergyNorm(root)
    p()

    print(p.v)
    print(p.get_v_mat(5))
    b = np.arange(d.n_free)
    print("b")
    print(b)
    print(p.compute_rom(b))
    a = np.arange(d.n_free ** 2).reshape((d.n_free, d.n_free))
    a = a + a.T
    print("a")
    print(a)
    print(p.compute_rom(a))
