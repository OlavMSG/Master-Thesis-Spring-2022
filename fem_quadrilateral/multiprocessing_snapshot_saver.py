# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from typing import Optional, Tuple, Callable, Union
from pathlib import Path

import numpy as np
import tqdm

import default_constants
import helpers
from .base_solver import BaseSolver
from matrix_lsq import Storage, DiskStorage, Snapshot
from itertools import product, repeat
import multiprocessing as mp


class MultiprocessingSnapshotSaver:
    storage: DiskStorage
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
        # assert len(self.storage) == 0

    def __call__(self, solver: BaseSolver, power_divider=3):
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
        ranges = np.array([self.geo_range, self.e_young_range, self.nu_poisson_range])
        mode_and_element = np.array([self.mode, solver.element])
        mls_order_and_llc = np.array([solver.mls_order, *solver.lower_left_corner])
        # save it
        root_mean = self.root / "mean"
        root_mean.mkdir(parents=True, exist_ok=True)
        # save a1, a2, f0, f1_dir, f2_dir for debugging.
        if solver.has_non_homo_dirichlet:
            Snapshot(root_mean, data_mean, a=a_mean,
                     p=solver.p, tri=solver.tri, edge=solver.edge,
                     dirichlet_edge=solver.dirichlet_edge,
                     neumann_edge=solver.neumann_edge,
                     grid_params=grid_params, ranges=ranges,
                     mode_and_element=mode_and_element, mls_order_and_llc=mls_order_and_llc,
                     a1=solver.a1, a2=solver.a2,
                     f0=solver.f0, rg=solver.rg,
                     f1_dir=solver.f1_dir, f2_dir=solver.f2_dir)
        else:
            Snapshot(root_mean, data_mean, a=a_mean,
                     p=solver.p, tri=solver.tri, edge=solver.edge,
                     dirichlet_edge=solver.dirichlet_edge,
                     neumann_edge=solver.neumann_edge,
                     grid_params=grid_params, ranges=ranges,
                     mode_and_element=mode_and_element, mls_order_and_llc=mls_order_and_llc,
                     a1=solver.a1, a2=solver.a2,
                     f0=solver.f0)

        geo_vec = helpers.get_vec_from_range(self.geo_range, self.geo_gird, self.mode)
        geo_mat = np.array(list(product(geo_vec, repeat=len(solver.sym_geo_params))))
        n_max = self.geo_gird ** len(solver.sym_geo_params)
        n_min = len(self.storage)
        # only use "1/power_divider power"
        num = max(mp.cpu_count() // power_divider, 1)
        it, r = divmod(n_max - n_min, num)
        for k in range(it):
            min_k = n_min + num * k
            max_k = min_k + num
            pool = mp.Pool(num, maxtasksperchild=1)
            for i in tqdm.tqdm(range(min_k, max_k), desc=f"Saving batch {k + 1} of {it + 1}."):
                input_i = [i, self.root, geo_mat[i, :], self.mode, self.material_grid, self.e_young_range,
                           self.nu_poisson_range, solver.mls_order, solver.solver_type, solver.n,
                           solver.input_f_func, solver.input_dirichlet_bc_func,
                           solver.input_get_dirichlet_edge_func, solver.input_neumann_bc_func,
                           solver.element, solver.lower_left_corner]
                pool.apply_async(one_snapshot_saver, input_i)
            pool.close()
            pool.join()
            del pool
            # check that storage is of rith length
            if (num_storage := len(self.storage)) != max_k:
                raise ValueError("Number of snapshots in storage does not match number of "
                                 f"snapshots that should have been saved. Number in storage {num_storage} v "
                                 f"{max_k}.")

        pool = mp.Pool(num, maxtasksperchild=1)
        for i in tqdm.tqdm(range(n_min + it * num, n_min + it * num + r), desc=f"Saving {it + 1} of {it + 1}"):
            input_i = [i, self.root, geo_mat[i, :], self.mode, self.material_grid, self.e_young_range,
                       self.nu_poisson_range, solver.mls_order, solver.solver_type, solver.n,
                       solver.input_f_func, solver.input_dirichlet_bc_func,
                       solver.input_get_dirichlet_edge_func, solver.input_neumann_bc_func,
                       solver.element, solver.lower_left_corner]
            pool.apply_async(one_snapshot_saver, input_i)
        pool.close()
        pool.join()
        del pool
        # check that storage is of rith length
        if (num_storage := len(self.storage)) != n_min + it * num + r:
            raise ValueError("Number of snapshots in storage does not match number of "
                             f"snapshots that should have been saved. Number in storage {num_storage} v "
                             f"{n_min + it * num + r}.")


def one_snapshot_saver(k: int, root: Path, geo_params: np.ndarray,
                       mode: str, material_grid: int, e_young_range: Tuple[float, float],
                       nu_poisson_range: Tuple[float, float], mls_order: int, solver_type: str, n: int,
                       f_func: Union[Callable, int], dirichlet_bc_func: Callable,
                       get_dirichlet_edge_func: Callable, neumann_bc_func: Callable,
                       element: str, lower_left_corner: Tuple[float, float]):
    e_young_vec = helpers.get_vec_from_range(e_young_range, material_grid, mode)
    nu_poisson_vec = helpers.get_vec_from_range(nu_poisson_range, material_grid, mode)

    # set up new solver.
    from .fem_quadrilateral_solvers import get_solver
    solver = get_solver(solver_type, n, f_func, dirichlet_bc_func, get_dirichlet_edge_func,
                        neumann_bc_func, element, *lower_left_corner)
    solver.matrix_lsq_setup(mls_order)
    solver.assemble(*geo_params)
    # compute solution and a-norm-squared for all e_young and nu_poisson
    # put in solution matrix and anorm2 vector
    s_mat = np.zeros((solver.n_free, material_grid ** 2))
    uh_anorm2_vec = np.zeros(material_grid ** 2)
    for i, (e_young, nu_poisson) in enumerate(product(e_young_vec, nu_poisson_vec)):
        solver.hfsolve(e_young, nu_poisson, print_info=False)
        s_mat[:, i] = solver.uh_free
        uh_anorm2_vec[i] = solver.uh_anorm2
    # matrix-LSQ data
    data = solver.mls_funcs(*geo_params).ravel()
    # save
    storage = DiskStorage(root)
    save_root = storage.root_of(k)
    save_root.mkdir(parents=True, exist_ok=True)
    if solver.has_non_homo_dirichlet:
        Snapshot(save_root, data, a1=solver.a1, a2=solver.a2,
                 f0=solver.f0,
                 f1_dir=solver.f1_dir, f2_dir=solver.f2_dir,
                 s_mat=s_mat, uh_anorm2=uh_anorm2_vec)

    else:
        Snapshot(save_root, data, a1=solver.a1, a2=solver.a2,
                 f0=solver.f0,
                 s_mat=s_mat, uh_anorm2=uh_anorm2_vec)
    print(f"saved in {save_root} for k={k}")
