# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
based on Specialization-Project-fall-2021
"""
from __future__ import annotations
from typing import Callable
import numpy as np

from .gauss_quadrature import line_integral_with_linear_basis
from ...helpers import expand_index


def assemble_f_neumann(n: int, p: np.ndarray, neumann_edge: np.ndarray,
                       neumann_bc_func: Callable, nq: int = 4) -> np.ndarray:
    """
    Assemble the neumann load vector

    Parameters
    ----------
    n : int
        number of node along one axis.
    p : np.ndarray
        list of points.
    neumann_edge : np.ndarray
        array of the edges of the triangulation.
    neumann_bc_func : Callable
        the neumann boundary condition function.
    nq : int, optional
        quadrature scheme order. The default is 4.

    Returns
    -------
    f_load_neumann : np.ndarray
        load vector for neumann.

    """
    n2d = n * n * 2
    # load vector
    f_load_neumann = np.zeros(n2d, dtype=float)
    for ek in neumann_edge:
        # p1 = p[ek[0], :]
        # p2 = p[ek[1], :]
        # expand the index
        expanded_ek = expand_index(ek)
        # add local contribution
        f_load_neumann[expanded_ek] += line_integral_with_linear_basis(*p[ek, :], g=neumann_bc_func, nq=nq)
    return f_load_neumann
