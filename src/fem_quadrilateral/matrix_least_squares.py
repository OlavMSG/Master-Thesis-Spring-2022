# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union, Iterable
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm as spnorm
from matrix_lsq import LeastSquares, Storage, DiskStorage

Matrix = Union[np.ndarray, sp.spmatrix]


def norm(mat: Matrix) -> float:
    # Frobenius norm of matrices
    # 2-norm of vectors
    if isinstance(mat, sp.spmatrix):
        return spnorm(mat)
    else:
        return np.linalg.norm(mat)


def mls_compute_from_fit(data: np.ndarray, mats: List[Matrix], drop: Union[Iterable[int], int] = None) -> Matrix:
    if drop is None:
        drop = []
    elif isinstance(drop, int):
        drop = [drop]
    if not all(isinstance(drop_i, int) for drop_i in drop):
        raise ValueError("All indexes in drop must be integers.")
    if isinstance(mats[0], sp.spmatrix):
        out = sp.csr_matrix(mats[0].shape, dtype=float)
    else:
        out = np.zeros_like(mats[0], dtype=float)
    for i, coeff in enumerate(data.ravel()):
        if i not in drop:
            out += coeff * mats[i]
    return out


class MatrixLSQ:
    storage: Storage
    root: Path

    a1_list: List[Matrix] = None
    a2_list: List[Matrix] = None
    f0_list: List[Matrix] = None
    f1_dir_list: List[Matrix] = None
    f2_dir_list: List[Matrix] = None
    num_kept: int = None
    drop: np.ndarray

    def __init__(self, root: Path):
        self.root = root
        self.storage = DiskStorage(root)
        self.fitter = LeastSquares(self.storage)

        self.has_non_homo_dirichlet = (self.root / "object-0" / "obj-f1_dir.npy").exists()

    def __call__(self):
        self.a1_list = self.fitter("a1")
        self.a2_list = self.fitter("a2")
        self.f0_list = self.fitter("f0")
        if self.has_non_homo_dirichlet:
            self.f1_dir_list = self.fitter("f1_dir")
            self.f2_dir_list = self.fitter("f2_dir")

        self.num_kept = len(self.a1_list)
