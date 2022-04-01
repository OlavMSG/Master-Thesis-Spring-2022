# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations

from itertools import product
from typing import Optional, Union
from pathlib import Path

import helpers
from .base_solver import BaseSolver

import scipy.sparse as sp
import tqdm
from scipy.linalg import eigh, fractional_matrix_power

import default_constants
from matrix_lsq import Storage, DiskStorage, Snapshot
import numpy as np

Matrix = Union[np.ndarray, sp.spmatrix]


class RbErrorComputer:
    root: Path
    storage: Storage

    def __init__(self, root: Path):
        self.root = root
        self.storage = DiskStorage(root)

        assert len(self.storage) != 0

    def __call__(self, solver: BaseSolver, e_young: float, nu_poisson: float, *geo_params: float,
                 n_rom: Optional[int] = None):
        # check if e_young, nu_poisson and geo_params matches an uh saved,
        # get data or hfsolve system, then
        # rbsolve system
        root_mean = self.root / "mean"
        mean_snapshot = Snapshot(root_mean)
        geo_gird, material_grid, num_geo_param = mean_snapshot["grid_params"]
        assert len(geo_params) == len(solver.sym_geo_params) == num_geo_param
        geo_range, e_young_range, nu_poisson_range = mean_snapshot["ranges"]
        mode = mean_snapshot["mode_and_element"][0]
        geo_vec = helpers.get_vec_from_range(geo_range, geo_gird, mode)
        print(geo_vec)
        # check if geo_params exist in saves and get index "via small hack"...
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
                # rbsolve system
                solver.rbsolve(e_young, nu_poisson, *geo_params, n_rom=n_rom, print_info=False)

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

                return err_anorm2, uh_anorm2


        # hfsolve
        if solver.is_assembled_and_free_from_root:
            raise ValueError("Solver comes from root, can not compute error; missing true a-matrix.")
        print("solving hf")
        # assemble system for geo_params
        solver.assemble(*geo_params)
        solver.hfsolve(e_young, nu_poisson, print_info=False)
        uh_anorm2 = solver.uh_anorm2
        # rbsolve system
        solver.rbsolve(e_young, nu_poisson, *geo_params, n_rom=n_rom, print_info=False)

        print(solver.uh_free.shape)
        print(solver.uh_rom_free.shape)
        print(uh_anorm2, np.sqrt(uh_anorm2))
        # compute error
        err = solver.uh_free - solver.uh_rom_free
        # compute a-norm
        err_anorm2 = err.T @ helpers.compute_a(e_young, nu_poisson, solver.a1, solver.a2) @ err
        print(err_anorm2)
        print(np.sqrt(err_anorm2) / np.sqrt(uh_anorm2))
        print(solver.uh_free.reshape(solver.uh_free.shape[0] // 2, 2))
        print(solver.uh_rom_free.reshape(solver.uh_free.shape[0] // 2, 2))

        return err_anorm2, uh_anorm2


class HfErrorComputer:
    root: Path
    storage: Storage

    def __init__(self, root: Path):
        self.root = root
        self.storage = DiskStorage(root)

        assert len(self.storage) != 0

    def __call__(self, solver: BaseSolver, e_young: float, nu_poisson: float, *geo_params: float):
        # check if e_young, nu_poisson and geo_params matches an uh saved,
        # get data or hfsolve system, then
        # hfsolve system with geo_params
        root_mean = self.root / "mean"
        mean_snapshot = Snapshot(root_mean)
        geo_gird, material_grid, num_geo_param = mean_snapshot["grid_params"]
        assert len(geo_params) == len(solver.sym_geo_params) == num_geo_param
        geo_range, e_young_range, nu_poisson_range = mean_snapshot["ranges"]
        mode = mean_snapshot["mode_and_element"][0]
        geo_vec = helpers.get_vec_from_range(geo_range, geo_gird, mode)
        print(geo_vec)
        # check if geo_params exist in saves and get index "via small hack"...
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
                # get true hf data
                snapshot = self.storage[index_geo]
                true_a1 = snapshot["a1"]
                true_a2 = snapshot["a2"]
                true_uh_free = snapshot["s_mat"][:, index_e_nu]
                true_uh_anorm2 = snapshot["uh_anorm2"][index_e_nu]
                # hfsolve system with geo_params
                solver.hfsolve(e_young, nu_poisson, *geo_params, print_info=False)

                print(true_uh_free.shape)
                print(solver.uh_free.shape)
                print(true_uh_anorm2)
                # compute error
                err = true_uh_free - solver.uh_free
                # compute a-norm
                err_anorm2 = err.T @ helpers.compute_a(e_young, nu_poisson, true_a1, true_a2) @ err
                print(err_anorm2)
                print(np.sqrt(err_anorm2) / np.sqrt(true_uh_anorm2))
                print(true_uh_free.reshape(true_uh_free.shape[0] // 2, 2))
                print(solver.uh_free.reshape(true_uh_free.shape[0] // 2, 2))
                return err_anorm2, true_uh_anorm2

        # hfsolve
        if solver.is_assembled_and_free_from_root:
            raise ValueError("Solver comes from root, can not compute error; missing true a-matrix.")
        print("solving hf")
        # assemble system for geo_params
        solver.assemble(*geo_params)
        solver.hfsolve(e_young, nu_poisson, print_info=False)
        # save true hf data
        true_a1 = solver.a1
        true_a2 = solver.a2
        true_uh_free = solver.uh_free
        true_uh_anorm2 = solver.uh_anorm2
        # hfsolve system with geo_params
        solver.hfsolve(e_young, nu_poisson, *geo_params, print_info=False)

        print(true_uh_free.shape)
        print(solver.uh_free.shape)
        print(true_uh_anorm2)
        # compute error
        err = true_uh_free - solver.uh_free
        # compute a-norm
        err_anorm2 = err.T @ helpers.compute_a(e_young, nu_poisson, true_a1, true_a2) @ err
        print(err_anorm2)
        print(np.sqrt(err_anorm2) / np.sqrt(true_uh_anorm2))
        print(true_uh_free.reshape(true_uh_free.shape[0] // 2, 2))
        print(solver.uh_free.reshape(true_uh_free.shape[0] // 2, 2))
        return err_anorm2, true_uh_anorm2

