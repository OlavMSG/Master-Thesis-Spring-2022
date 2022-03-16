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
        ...

if __name__ == '__main__':
    ...
