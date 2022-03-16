# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations

from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import tqdm

import default_constants
import helpers
from .base_solver import BaseSolver
from matrix_lsq import Storage, DiskStorage, Snapshot
from itertools import product, repeat


class SnapshotSaver:
    storage: Storage
    root: Path
    geo_gird: int
    geo_range: Tuple[float, float]
    mode: str
    material_grid: int = default_constants.e_nu_grid
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

        # for now
        assert len(self.storage) == 0

    def __call__(self, solver: BaseSolver):
        if not solver.mls_has_been_setup:
            solver.matrix_lsq_setup()

        # save the mean as a special Snapshot.
        geo_mean = np.mean(self.geo_range)
        e_mean = np.mean(self.e_young_range)
        nu_mean = np.mean(self.nu_poisson_range)
        # make the data
        solver.assemble(*repeat(geo_mean, len(solver.sym_geo_params)))
        data_mean = solver.mls_funcs(*repeat(geo_mean, len(solver.sym_geo_params))).ravel()
        a_mean = helpers.compute_a(e_mean, nu_mean, solver.a1, solver.a2)
        # for now save
        grid_params = np.array([self.geo_gird, self.material_grid, len(solver.sym_geo_params)])
        # save it
        root_mean = self.root / "mean"
        root_mean.mkdir(parents=True, exist_ok=True)
        # save a1, a2, f0, f1_dir, f2_dir for debugging.
        if solver.has_non_homo_dirichlet:
            Snapshot(root_mean, data_mean, a=a_mean,
                     p=solver.p, tri=solver.tri, edge=solver.edge,
                     dirichlet_edge=solver.dirichlet_edge,
                     neumann_edge=solver.neumann_edge,
                     grid_params=grid_params,
                     a1=solver.a1, a2=solver.a2,
                     f0=solver.f0,
                     f1_dir=solver.f1_dir, f2_dir=solver.f2_dir)
        else:
            Snapshot(root_mean, data_mean, a=a_mean,
                     p=solver.p, tri=solver.tri, edge=solver.edge,
                     dirichlet_edge=solver.dirichlet_edge,
                     neumann_edge=solver.neumann_edge,
                     grid_params=grid_params,
                     a1=solver.a1, a2=solver.a2,
                     f0=solver.f0)

        # get all vectors from ranges
        geo_vec = helpers.get_vec_from_range(self.geo_range, self.geo_gird, self.mode)
        e_young_vec = helpers.get_vec_from_range(self.e_young_range, self.material_grid, self.mode)
        nu_poisson_vec = helpers.get_vec_from_range(self.nu_poisson_range, self.material_grid, self.mode)
        # make snapshots
        for geo_params in tqdm.tqdm(product(geo_vec, repeat=len(solver.sym_geo_params)), desc="Saving"):
            solver.assemble(*geo_params)
            # compute solution and a-norm-squared for all e_young and nu_poisson
            # put in solution matrix and anorm2 vector
            s_mat = np.zeros((solver.n_free, self.material_grid ** 2))
            uh_anorm2_vec = np.zeros(self.material_grid ** 2)
            for i, (e_young, nu_poisson) in enumerate(product(e_young_vec, nu_poisson_vec)):
                solver.hfsolve(e_young, nu_poisson, print_info=False)
                s_mat[:, i] = solver.uh_free
                uh_anorm2_vec[i] = solver.uh_anorm2
            # matrix-LSQ data
            data = solver.mls_funcs(*geo_params).ravel()
            # save
            if solver.has_non_homo_dirichlet:
                self.storage.append(data, a1=solver.a1, a2=solver.a2,
                                    f0=solver.f0,
                                    f1_dir=solver.f1_dir, f2_dir=solver.f2_dir,
                                    s_mat=s_mat, uh_anorm2=uh_anorm2_vec)

            else:
                self.storage.append(data, a1=solver.a1, a2=solver.a2,
                                    f0=solver.f0,
                                    s_mat=s_mat, uh_anorm2=uh_anorm2_vec)

    def get_parameter_matrix(self, solver: BaseSolver) -> np.ndarray:
        geo_vec = helpers.get_vec_from_range(self.geo_range, self.geo_gird, self.mode)
        e_young_vec = helpers.get_vec_from_range(self.e_young_range, self.material_grid, self.mode)
        nu_poisson_vec = helpers.get_vec_from_range(self.nu_poisson_range, self.material_grid, self.mode)
        return np.array(list(product(*repeat(geo_vec, len(solver.sym_geo_params)), e_young_vec, nu_poisson_vec)))

    def vipe(self):
        user_input = str(input(f"Do you really want to vipe storage in \'{self.root}\'? This will delete all the "
                               f"files stored. y: "))
        if user_input.lower() == "y":
            root_mean = self.root / "mean"
            for path in root_mean.glob('*'):
                path.unlink()
            root_mean.rmdir()
            self.storage.vipe(user_confirm=False)


if __name__ == '__main__':
    from fem_quadrilateral import DraggableCornerRectangleSolver

    d = DraggableCornerRectangleSolver(3, 0)
    root = Path("test_storage1")
    s = SnapshotSaver(root, 3, d.geo_param_range, material_grid=3)
    # print(s.get_parameter_matrix(d))
    s(d)

    # s.vipe()
