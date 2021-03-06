# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

import scipy.sparse as sp
import tqdm
from matplotlib import pyplot as plt
from scipy.linalg import eigh, fractional_matrix_power

from . import default_constants
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
    ns: int = None

    def __init__(self, root: Path, eps_pod: Optional[float] = None):
        self.root = root
        self.storage = DiskStorage(root)
        if eps_pod is not None:
            self.eps_pod = eps_pod

        assert len(self.storage) != 0

    def __call__(self):
        # NOTE
        # known limitations:
        # eigh - scipy.linalg.eigh is not compliantly stable, stable enough here
        #      - it can also be quite slow
        # fractional_matrix_power - scipy.linalg.fractional_matrix_power is really slow (is in else)
        #                         - at least slower than eigh
        #                         - sparsity of a_mean is lost in input where a_mean.A is called giving the np.array
        #                         - function can use much RAM if a_mean is large ~ 10_000 x 10_000
        # testing if case against each other on case with n_free = 12_960 and ns = 15_625
        #                         - gives 3:44 in eigh for corr_mat (times in mm:ss)
        #                         - gives 20:02 in fractional_matrix_power and 2:09 in eigh for k_mat
        # if ns <= n_free - is not necessary because corr_mat and k_mat have the same eigenvalues
        #                - but it gives the smallest matrix between corr_mat and k_mat

        # get mean matrix
        root_mean = self.root / "mean"
        mean_snapshot = Snapshot(root_mean)
        a_mean = mean_snapshot["a"]
        n_free = a_mean.shape[0]
        geo_gird, material_grid, num_geo_param = mean_snapshot["grid_params"]
        m = material_grid ** 2
        self.ns = geo_gird ** num_geo_param * m
        # print(f"ns: {ns}, n_free: {n_free}, ns <= n_free: {ns <= n_free}.")
        s_mat = np.zeros((n_free, self.ns), dtype=float)

        for i, snapshot in tqdm.tqdm(enumerate(self.storage), desc="Loading data"):
            s_mat[:, i * m:(i + 1) * m] = snapshot["s_mat"]
        if (s_mat == 0).all():
            error_text = "Solution matrix is zero, can not compute POD for building a reduced model. " \
                         + "The most likely cause is f_func=0, dirichlet_bc_func=0 and neumann_bc_func=0, " \
                         + "where two last may be None."
            raise ValueError(error_text)

        if self.ns <= n_free:
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
            # build matrix
            k_mat = x05 @ s_mat @ s_mat.T @ x05
            # find the eigenvalues and eigenvectors of it
            self.sigma2_vec, zeta_mat = eigh(k_mat)
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
        if (n_rom is None) or (n_rom == self.n_rom):
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

    def plot_singular_values(self):
        if self.sigma2_vec is None:
            raise ValueError("No singular values available, call Pod first.")
        # set nice plotting
        fontsize = 20
        new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
                      'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
                      'figure.titlesize': fontsize,
                      'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
        plt.rcParams.update(new_params)
        plt.figure("Singular values")
        plt.title("Singular values, scaled to $\\sigma_1$")
        arg0 = np.argwhere(self.sigma2_vec >= 0)
        sigma_vec = np.sqrt(self.sigma2_vec[arg0])
        rel_sigma_vec = sigma_vec / sigma_vec[0]
        plt.semilogy(np.arange(len(rel_sigma_vec)) + 1, rel_sigma_vec, "mD-", label="Singular Values, $\\sigma_i$.")
        plt.xlabel("$i$")
        plt.ylabel("$\\sigma_i$")
        plt.grid()
        plt.legend()

    def plot_relative_information_content(self):
        if self.sigma2_vec is None:
            raise ValueError("No singular values available, call Pod first.")
        # set nice plotting
        fontsize = 20
        new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
                      'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
                      'figure.titlesize': fontsize,
                      'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
        plt.rcParams.update(new_params)
        arg0 = np.argwhere(self.sigma2_vec >= 0)
        i_n = np.cumsum(self.sigma2_vec[arg0]) / np.sum(self.sigma2_vec[arg0])
        plt.figure("Relative information content")
        plt.title("Relative information content, $I(N)$")
        plt.plot(np.arange(len(i_n)) + 1, i_n, "gD-")
        plt.plot(self.n_rom, i_n[self.n_rom - 1], "bo", label=f"$(N, I(N))=({self.n_rom}, {i_n[self.n_rom-1]:.5f})$")
        plt.xlabel("$N$")
        plt.ylabel("$I(N)$")
        plt.grid()
        plt.legend()
