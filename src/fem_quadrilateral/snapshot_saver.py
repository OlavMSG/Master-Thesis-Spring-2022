# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations

from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import tqdm

from . import default_constants, helpers
from .base_solver import BaseSolver
# we choose to not update to Compressed versions
from matrix_lsq import DiskStorage, Snapshot
from itertools import product, repeat
from scipy.stats.qmc import LatinHypercube


class SnapshotSaver:
    storage: DiskStorage
    root: Path
    geo_grid: int
    geo_range: Tuple[float, float]
    mode: str
    material_grid: int = default_constants.e_nu_grid
    e_young_range: Tuple[float, float] = default_constants.e_young_range
    nu_poisson_range: Tuple[float, float] = default_constants.nu_poisson_range

    def __init__(self, root: Path, geo_grid: int, geo_range: Tuple[float, float],
                 mode: str = "uniform",
                 material_grid: Optional[int] = None,
                 e_young_range: Optional[Tuple[float, float]] = None,
                 nu_poisson_range: Optional[Tuple[float, float]] = None,
                 use_latin_hypercube: bool = False, latin_hypercube_seed: Optional[int] = None):
        self.root = root
        self.storage = DiskStorage(root)
        self.geo_grid = geo_grid
        self.geo_range = geo_range
        self.mode = mode
        self.use_latin_hypercube = use_latin_hypercube
        self.latin_hypercube_seed = latin_hypercube_seed
        if material_grid is not None:
            self.material_grid = material_grid
        if e_young_range is not None:
            self.e_young_range = e_young_range
        if nu_poisson_range is not None:
            self.nu_poisson_range = nu_poisson_range

        # for now
        assert len(self.storage) == 0

    def __call__(self, solver: BaseSolver):
        # save the mean as a special Snapshot.
        geo_mean = np.mean(self.geo_range)
        e_mean = np.mean(self.e_young_range)
        nu_mean = np.mean(self.nu_poisson_range)
        # make the data
        solver.assemble(*repeat(geo_mean, len(solver.sym_geo_params)))
        data_mean = solver.mls_funcs(*repeat(geo_mean, len(solver.sym_geo_params))).ravel()
        a_mean = helpers.compute_a(e_mean, nu_mean, solver.a1, solver.a2)
        # for now save
        grid_params = np.array([self.geo_grid, self.material_grid, len(solver.sym_geo_params)])
        ranges = np.array([self.geo_range, self.e_young_range, self.nu_poisson_range])
        mode_and_element = np.array([self.mode, solver.element])
        mls_order_and_llc = np.array([solver.mls_order, *solver.lower_left_corner])
        solver_type = np.array([solver.solver_type])
        # save it
        root_mean = self.root / "mean"
        root_mean.mkdir(parents=True, exist_ok=True)
        if solver.has_non_homo_dirichlet:
            Snapshot(root_mean, data_mean, a=a_mean,
                     p=solver.p, tri=solver.tri, edge=solver.edge,
                     dirichlet_edge=solver.dirichlet_edge,
                     neumann_edge=solver.neumann_edge,
                     grid_params=grid_params, ranges=ranges, solver_type=solver_type,
                     mode_and_element=mode_and_element, mls_order_and_llc=mls_order_and_llc)
        else:
            Snapshot(root_mean, data_mean, a=a_mean,
                     p=solver.p, tri=solver.tri, edge=solver.edge,
                     dirichlet_edge=solver.dirichlet_edge,
                     neumann_edge=solver.neumann_edge,
                     grid_params=grid_params, ranges=ranges, solver_type=solver_type,
                     mode_and_element=mode_and_element, mls_order_and_llc=mls_order_and_llc)
        print(f"Saved mean in {root_mean}")

        # get all vectors from ranges
        if self.use_latin_hypercube:
            sampler = LatinHypercube(d=len(solver.sym_geo_params), seed=self.latin_hypercube_seed)
            geo_mat = 0.5 * ((self.geo_range[1] - self.geo_range[0]) * sampler.random(n=self.geo_grid)
                             + (self.geo_range[1] + self.geo_range[0]))
        else:
            geo_vec = helpers.get_vec_from_range(self.geo_range, self.geo_grid, self.mode)
            geo_mat = np.array(list(product(geo_vec, repeat=len(solver.sym_geo_params))))

        e_young_vec = helpers.get_vec_from_range(self.e_young_range, self.material_grid, self.mode)
        nu_poisson_vec = helpers.get_vec_from_range(self.nu_poisson_range, self.material_grid, self.mode)
        # make snapshots
        for geo_params in tqdm.tqdm(geo_mat, desc="Saving"):
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
        geo_vec = helpers.get_vec_from_range(self.geo_range, self.geo_grid, self.mode)
        e_young_vec = helpers.get_vec_from_range(self.e_young_range, self.material_grid, self.mode)
        nu_poisson_vec = helpers.get_vec_from_range(self.nu_poisson_range, self.material_grid, self.mode)
        return np.array(list(product(*repeat(geo_vec, len(solver.sym_geo_params)), e_young_vec, nu_poisson_vec)))
