# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations

from itertools import product
from typing import Optional, Union
from pathlib import Path

import helpers
from fem_quadrilateral.base_solver import BaseSolver

import scipy.sparse as sp
import tqdm
from scipy.linalg import eigh, fractional_matrix_power

import default_constants
from matrix_lsq import Storage, DiskStorage, Snapshot
import numpy as np

Matrix = Union[np.ndarray, sp.spmatrix]


class ErrorComputer:
    root: Path
    storage: Storage

    def __init__(self, root: Path):
        self.root = root
        self.storage = DiskStorage(root)

        assert len(self.storage) != 0

    def __call__(self, solver: BaseSolver, e_young: float, nu_poisson: float, *geo_params: float):
        # assume that uh_rom exists, and uses e_young, nu_poisson and geo_params used for call here
        # check if e_young, nu_poisson and geo_params matches an uh saved.

        root_mean = self.root / "mean"
        mean_snapshot = Snapshot(root_mean)
        geo_gird, material_grid, num_geo_param = mean_snapshot["grid_params"]
        assert len(geo_params) == len(r.sym_geo_params) == num_geo_param
        geo_range, e_young_range, nu_poisson_range = mean_snapshot["ranges"]
        mode = mean_snapshot["mode"][0]
        m = material_grid ** 2
        ns = geo_gird ** num_geo_param * m

        geo_vec = helpers.get_vec_from_range(geo_range, geo_gird, mode)
        print(geo_vec)
        # check if geo_params exist in saves
        # and get index "via small hack"...
        if any((index_geo := j) == j and
               all(abs(geo_params[i] - check_geo_params[i]) < default_constants.default_tol for i in
                   range(num_geo_param)) for j, check_geo_params in
               enumerate(product(geo_vec, repeat=len(solver.sym_geo_params)))):
            e_young_vec = helpers.get_vec_from_range(e_young_range, material_grid, mode)
            nu_poisson_vec = helpers.get_vec_from_range(nu_poisson_range, material_grid, mode)
            # check if e_young and nu_poisson exist in saves
            # and get index "via small hack"...
            if any((index_e_nu := i) == i and (abs(e_young - check_e_young) < default_constants.default_tol) and (
                    abs(nu_poisson - check_nu_poisson) < default_constants.default_tol) for
                   i, (check_e_young, check_nu_poisson) in enumerate(product(e_young_vec, nu_poisson_vec))):
                # get uh data
                snapshot = self.storage[index_geo]
                a1 = snapshot["a1"]
                a2 = snapshot["a2"]
                uh_free = snapshot["s_mat"][:, index_e_nu]
                uh_anorm2 = snapshot["uh_anorm2"][index_e_nu]
                print(uh_free.shape)
                print(solver.uh_rom_free.shape)
                print(uh_anorm2)
                # compute error
                err = uh_free - solver.uh_rom_free
                # compute a-norm
                err_anorm2 = err.T @ helpers.compute_a(e_young, nu_poisson, a1, a2) @ err
                print(err_anorm2)
                print(np.sqrt(err_anorm2) / np.sqrt(uh_anorm2))
                print(uh_free.reshape(uh_free.shape[0] // 2, 2))
                print(solver.uh_rom_free.reshape(uh_free.shape[0] // 2, 2))


        """for geo_params in tqdm.tqdm(), desc="Saving"):
            solver.assemble(*geo_params)
            # compute solution and a-norm-squared for all e_young and nu_poisson
            # put in solution matrix and anorm2 vector
            s_mat = np.zeros((solver.n_free, self.material_grid ** 2))
            uh_anorm2_vec = np.zeros(self.material_grid ** 2)
            for i, (e_young, nu_poisson) in enumerate(product(e_young_vec, nu_poisson_vec)):"""


if __name__ == '__main__':
    from fem_quadrilateral import ScalableRectangleSolver

    root = Path("test_storage1")
    err = ErrorComputer(root)


    def dir_bc(x, y):
        return x, 0


    r = ScalableRectangleSolver(2, 0, dirichlet_bc_func=dir_bc)
    e_mean = np.mean(default_constants.e_young_range)
    nu_mean = np.mean(default_constants.nu_poisson_range)
    geo_mean = np.mean(r.geo_param_range)
    r.assemble(geo_mean, geo_mean)
    r.matrix_lsq_setup(2)

    r.build_rb_model(root)
    r.rbsolve(e_mean, nu_mean, geo_mean, geo_mean)
    err(r, e_mean, nu_mean, geo_mean, geo_mean)
