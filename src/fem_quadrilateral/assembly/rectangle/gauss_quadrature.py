# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
from Specialization-Project-fall-2021
"""
from __future__ import annotations
from typing import Callable, Union, Tuple, Optional
import numpy as np
from scipy.special import roots_legendre
from itertools import product


def minus_one_one_to_r2(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray,
                        z: np.ndarray) -> np.ndarray:
    # mapping (xhi1, xhi2) to (x, y) given matrix of xhi-s Z
    return 0.25 * (np.multiply.outer(p1, (1 - z[:, 0]) * (1 - z[:, 1]))
                   + np.multiply.outer(p2, (1 + z[:, 0]) * (1 - z[:, 1]))
                   + np.multiply.outer(p3, (1 + z[:, 0]) * (1 + z[:, 1]))
                   + np.multiply.outer(p4, (1 - z[:, 0]) * (1 + z[:, 1])))


def det_jac_minus_one_one_to_r2(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray,
                                z: np.ndarray) -> np.ndarray:
    jac0 = ((p1 - p2 + p3 - p4) * z - p1 + p2 + p3 - p4).T
    jac1 = (p1 * (z - 1) + z * (-p2 + p3 - p4) - p2 + p3 + p4).T
    return 0.0625 * (jac0[0, :] * jac1[1, :] - jac1[0, :] * jac0[1, :])


def get_points_and_weights_quad_2D(nq_x: int, nq_y: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get Gauss quadrature points and weighs in 2D

    Parameters
    ----------
    nq_x : int
        scheme order in x.
    nq_y : int
        scheme order in y.


    Returns
    -------
    z : np.ndarray
        quadrature points.
    rho : np.ndarray
        quadrature weights.

    """
    z_x, rho_x = roots_legendre(nq_x)
    z_y, rho_y = roots_legendre(nq_y)

    z = np.array(list(product(z_x, z_y)))
    rho = np.kron(rho_x, rho_y)
    return z, rho


def quadrature2D(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray,
                 g: Callable, nq_x: int, nq_y: Optional[int] = None) -> np.ndarray:
    """
    Integral over the triangle with vertices in p1, p2 and p3

    Parameters
    ----------
    p1 : np.ndarray
        point p1.
    p2 : np.ndarray
        point p2.
    p3 : np.ndarray
        point p3.
    p4 : np.ndarray
        point p4.
    g : function
        the function to integrate.
    nq_x : int
        scheme order in x.
    nq_y : int, optional
        scheme order in y, equal to nq_x if None. Default None.

    Returns
    -------
    np.ndarray
        value of integral.

    """
    if nq_y is None:
        nq_y = nq_x
    # Weights and gaussian quadrature points
    z, rho = get_points_and_weights_quad_2D(nq_x, nq_y)

    # Calculating the Gaussian quadrature summation formula
    return np.sum(rho * g(*minus_one_one_to_r2(p1, p2, p3, p4, z)) * det_jac_minus_one_one_to_r2(p1, p2, p3, p4, z))


def quadrature2D_vector(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray,
                 g: Callable, nq_x: int, nq_y: Optional[int] = None) -> np.ndarray:
    """
    Integrals over the triangle with vertices in p1, p2 and p3, assuming that g = [g1, g2], return int g1 and int g2

    Parameters
    ----------
    p1 : np.ndarray
        point p1.
    p2 : np.ndarray
        point p2.
    p3 : np.ndarray
        point p3.
    p4 : np.ndarray
        point p4.
    g : function
        the vector function g = [g1, g2] to integrate.
    nq_x : int
        scheme order in x.
    nq_y : int, optional
        scheme order in y, equal to nq_x if None. Default None.

    Returns
    -------
    np.ndarray
        values of the integrals' int g1 and int g2.

    """
    if nq_y is None:
        nq_y = nq_x
    # Weights and gaussian quadrature points
    z, rho = get_points_and_weights_quad_2D(nq_x, nq_y)

    # Calculating the Gaussian quadrature summation formula
    return np.sum(rho * g(*minus_one_one_to_r2(p1, p2, p3, p4, z)) * det_jac_minus_one_one_to_r2(p1, p2, p3, p4, z),
                  axis=1)


def get_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Get the area of the rectangle with vertices in p1, p2, p3 and p4

    Parameters
    ----------
    p1 : np.ndarray
        point p1.
    p2 : np.ndarray
        point p2.
    p3 : np.ndarray
        point p3.
    p4 : np.ndarray
        point p4.

    Returns
    -------
    float
        area of rectangle.

    """
    return 0.5 * abs((p1[0] - p3[0]) * (p2[1] - p4[1]) - (p2[0] - p4[0]) * (p1[1] - p3[0]))
