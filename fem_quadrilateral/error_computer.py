# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations

from typing import Optional, Union
from pathlib import Path
from base_solver import BaseSolver

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
        root_mean = self.root / "mean"
        mean_snapshot = Snapshot(root_mean)
        geo_gird, material_grid, num_geo_param = mean_snapshot["grid_params"]
        m = material_grid ** 2
        ns = geo_gird ** num_geo_param * m
        error2_vec = np.zeros(ns, dtype=float)



if __name__ == '__main__':
    ...
