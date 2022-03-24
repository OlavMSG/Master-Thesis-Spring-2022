# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Optional, List, Tuple, Callable, Union
from importlib.util import find_spec
import numpy as np
import scipy.sparse as sp

symengine_is_found = (find_spec("symengine") is not None)
if symengine_is_found:
    import symengine as sym
else:
    import sympy as sym


class BaseSolver(Protocol):
    ref_plate: Tuple[int, int]
    implemented_elements: List[str]
    sym_phi: sym.Matrix
    sym_params: sym.Matrix
    sym_geo_params: sym.Matrix
    sym_mls_funcs: sym.Matrix
    n_free: int
    mls_has_been_setup: bool
    mls_funcs: Callable
    has_non_homo_dirichlet: bool
    a1: sp.spmatrix
    a2: sp.spmatrix
    f0: np.ndarray
    f1_dir: np.ndarray
    f2_dir: np.ndarray
    # a1_dir: sp.spmatrix
    # a2_dir: sp.spmatrix
    p: np.ndarray
    tri: np.ndarray
    edge: np.ndarray
    dirichlet_edge: np.ndarray
    neumann_edge: np.ndarray

    def set_quadrature_scheme_order(self, nq: int, nq_y: Optional[int] = None):
        ...

    def hfsolve(self, e_young: float, nu_poisson: float, *geo_params: Optional[float], print_info: bool = True):
        ...

    def rbsolve(self, e_young: float, nu_poisson: float, *geo_params: float, n_rom: Optional[int] = None,
                print_info: bool = True):
        ...

    def save_snapshots(self, root: Path, geo_grid: int,
                       geo_range: Tuple[float, float] = None,
                       mode: str = "uniform",
                       material_grid: Optional[int] = None,
                       e_young_range: Optional[Tuple[float, float]] = None,
                       nu_poisson_range: Optional[Tuple[float, float]] = None):
        ...

    def matrix_lsq_setup(self, mls_order: int = 1):
        ...

    def matrix_lsq(self, root: Path):
        ...

    def build_rb_model(self, root: Path, eps_pod: Optional[float] = None):
        ...

    def assemble(self, mu1: float, mu2: float, mu3: float, mu4: float, mu5: float, mu6: float, mu7: float, mu8: float):
        ...

    @property
    def mls_num_kept(self) -> int:
        return -1

    @property
    def n_rom(self) -> int:
        return -1

    @property
    def n_rom_max(self) -> int:
        return -1

    @property
    def uh_free(self) -> np.ndarray:
        return np.ndarray()

    @property
    def uh_full(self) -> np.ndarray:
        return np.ndarray()

    @property
    def uh_anorm2(self) -> np.ndarray:
        return np.ndarray()

    @property
    def uh_rom_free(self) -> np.ndarray:
        return np.ndarray()

    @property
    def uh_rom_full(self) -> np.ndarray:
        return np.ndarray()
