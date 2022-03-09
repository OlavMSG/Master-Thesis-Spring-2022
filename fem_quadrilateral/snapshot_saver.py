# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations

from typing import Optional, Tuple
from pathlib import Path

import numpy as np

import default_constants
import helpers
from base_solver import BaseSolver
from matrix_lsq import Storage, DiskStorage
from itertools import product


class SnapshotSaver:
    storage: Storage
    root: Path
    geo_gird: int
    geo_range: Tuple[float, float]
    material_grid: int = default_constants.e_nu_grid
    mode: str
    e_young_range: Tuple[float, float] = default_constants.e_young_range
    nu_poisson_range: Tuple[float, float] = default_constants.nu_poisson_range

    def __init__(self, root: Path, geo_grid: int, geo_range: Tuple[float, float],
                 mode: str = "uniform",
                 material_grid: Optional[int] = None,
                 e_young_range: Optional[Tuple[float, float]] = None,
                 nu_poisson_range: Optional[Tuple[float, float]] = None):
        self.root = root
        self.storage = DiskStorage(root)
        self.geo_gird = geo_grid
        self.geo_range = geo_range
        self.mode = mode
        if material_grid is not None:
            self.material_grid = material_grid
        if e_young_range is not None:
            self.e_young_range = e_young_range
        if nu_poisson_range is not None:
            self.nu_poisson_range = nu_poisson_range

    def __call__(self, solver: BaseSolver):
        if not solver.mls_has_been_setup:
            solver.mls_setup()
        geo_vec = helpers.get_vec_from_range(self.geo_range, self.geo_gird, self.mode)
        e_young_vec = helpers.get_vec_from_range(self.e_young_range, self.material_grid, self.mode)
        nu_poisson_vec = helpers.get_vec_from_range(self.nu_poisson_range, self.material_grid, self.mode)
        geo_vecs = [geo_vec] * len(solver.sym_geo_params)
        for geo_params in product(*geo_vecs):
            solver.assemble(*geo_params)
            s_mat = np.zeros(solver.n_free)
            for i, e_young, nu_poisson in enumerate(product(e_young_vec, nu_poisson_vec)):
                solver.hfsolve(e_young, nu_poisson, print_info=False)
                s_mat[i, :] = solver.uh_free
            data = solver.mls_funcs(*geo_params)
            if solver.has_non_homo_dirichlet:
                self.storage.append(data, a1=solver.a1, a2=solver.a2,
                                    f0=solver.f0,
                                    f1_dir=solver.f1_dir, f2_dir=solver.f2_dir,
                                    s_mat=s_mat)
            else:
                self.storage.append(data, a1=solver.a1, a2=solver.a2,
                                    f0=solver.f0,
                                    s_mat=s_mat)

    def get_parameter_matrix(self, solver: BaseSolver) -> np.ndarray:
        geo_vec = helpers.get_vec_from_range(self.geo_range, self.geo_gird, self.mode)
        e_young_vec = helpers.get_vec_from_range(self.e_young_range, self.material_grid, self.mode)
        nu_poisson_vec = helpers.get_vec_from_range(self.nu_poisson_range, self.material_grid, self.mode)
        geo_vecs = [geo_vec] * len(solver.sym_geo_params)
        return np.array(list(product(*geo_vecs, e_young_vec, nu_poisson_vec)))


if __name__ == '__main__':
    from fem_quadrilateral import DraggableCornerRectangleSolver

    d = DraggableCornerRectangleSolver(3, 0)
    root = Path("test_storage")
    s = SnapshotSaver(root, 3, d.geo_param_range)
    print(s.get_parameter_matrix(d))
