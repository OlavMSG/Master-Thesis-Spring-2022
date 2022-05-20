# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
from Specialization-Project-fall-2021
"""
from __future__ import annotations
from typing import Callable
import numpy as np
from scipy.special import roots_legendre


def line_integral_with_linear_basis(a: np.ndarray, b: np.ndarray, g: Callable, nq: int) -> np.ndarray:
    """
    Line integral with local triangle basis functions on the line from a to b

    Parameters
    ----------
    a : np.ndarray
        point a.
    b : np.ndarray
        point b.
    g : Callable
        the function to integrate (times basis functions).
    nq : int
        scheme order.

    Returns
    -------
    line_ints : np.ndarray
        array of line integrals.

    """
    z, rho = roots_legendre(nq)

    # parameterization of the line L between a and b,
    # r(t) = (1-t)a/2 + (1+t)b/2
    z0 = 0.5 * (1.0 - z)
    z1 = 0.5 * (1.0 + z)
    xy = np.multiply.outer(a, z0) + np.multiply.outer(b, z1)

    # r'(t) = -a/2+ b/2 = (b-a)/2, |r'(t)| = norm(b-a)
    abs_r_t = 0.5 * np.linalg.norm(b - a, ord=2)
    # evaluate g
    g_vec_x, g_vec_y = np.hsplit(g(*xy), 2)

    # compute the integrals numerically
    line_ints = np.zeros(4, dtype=float)
    # g times local basis for a
    line_ints[0] = abs_r_t * np.sum(rho * g_vec_x * z0)
    line_ints[1] = abs_r_t * np.sum(rho * g_vec_y * z0)
    # g times local basis for b
    line_ints[2] = abs_r_t * np.sum(rho * g_vec_x * z1)
    line_ints[3] = abs_r_t * np.sum(rho * g_vec_y * z1)
    return line_ints
