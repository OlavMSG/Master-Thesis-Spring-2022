# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
from pathlib import Path

from scipy.linalg import eigh, fractional_matrix_power

import default_constants
from matrix_lsq import Storage, DiskStorage, Snapshot
import numpy as np


class PodWithEnergyNorm:
    root: Path
    storage: Storage
    eps_pod: float = default_constants.eps_pod
    n_max: int
    n_rom: int
    sigma2_vec: np.ndarray
    i_n: np.ndarray
    v_mat_n_max: np.ndarray
    v: np.ndarray

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

        for i, snapshot in enumerate(self.storage):
            s_mat[:, i * m:(i + 1) * m] = snapshot["s_mat"]

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

            self.n_max = np.min(np.argwhere(self.sigma2_vec < 0)) + 1
            self.v_mat_n_max = s_mat @ zeta_mat[:, :self.n_max] / np.sqrt(self.sigma2_vec[:self.n_max])

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

            self.n_max = np.min(np.argwhere(self.sigma2_vec < 0)) + 1
            self.v_mat_n_max = np.linalg.solve(x05, zeta_mat[:, :self.n_max])

        self.v = self.v_mat_n_max[:, :self.n_rom]

    def get_v_mat(self, n_rom: int) -> np.ndarray:
        if n_rom > self.n_max:
            raise ValueError(f"No V matrix available for n_rom={n_rom}, max n_rom is {self.n_max}.")
        else:
            return self.v_mat_n_max[:, :n_rom]


if __name__ == '__main__':
    from fem_quadrilateral import DraggableCornerRectangleSolver


    def dir_bc(x, y):
        return x, 0


    n = 2
    root = Path("test_storage2")

    if len(DiskStorage(root)) == 0:
        d = DraggableCornerRectangleSolver(3, 0, dirichlet_bc_func=dir_bc)
        d.save_snapshots(root, 3)

    p = PodWithEnergyNorm(root)
    p()

    print(p.v)
