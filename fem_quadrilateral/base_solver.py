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

Matrix = Union[np.ndarray, sp.spmatrix]


class BaseSolver(Protocol):
    ref_plate: Tuple[int, int]
    implemented_elements: List[str]
    sym_phi: sym.Matrix
    sym_params: sym.Matrix
    sym_geo_params: sym.Matrix
    n_free: int
    mls_has_been_setup: bool
    mls_funcs: Callable
    has_non_homo_dirichlet: bool
    a1: Matrix
    a2: Matrix
    f0: Matrix
    f1_dir: Matrix
    f2_dir: Matrix
    p: Matrix
    tri: Matrix
    edge: Matrix
    dirichlet_edge: Matrix
    neumann_edge: Matrix

    def mls_setup(self, mls_order: int = 1, use_negative_mls_order: bool = False):
        ...

    def set_quadrature_scheme_order(self, nq: int, nq_y: Optional[int] = None):
        ...

    def set_geo_param_range(self, start: float, stop: float):
        ...

    def hfsolve(self, e_young: float, nu_poisson: float, print_info: bool = True):
        ...

    def save_snapshots(self, root: Path, geo_grid: int,
                       mode: str = "uniform",
                       material_grid: Optional[int] = None,
                       e_young_range: Optional[Tuple[float, float]] = None,
                       nu_poisson_range: Optional[Tuple[float, float]] = None):
        ...

    def assemble(self, mu1: float, mu2: float, mu3: float, mu4: float, mu5: float, mu6: float, mu7: float, mu8: float):
        ...

    @property
    def uh_free(self) -> np.ndarray:
        return np.ndarray()

    @property
    def uh_full(self) -> np.ndarray:
        return np.ndarray()

    @property
    def uh_anorm2(self) -> np.ndarray:
        return np.ndarray()
