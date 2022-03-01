# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
from Specialization-Project-fall-2021
"""

import numpy as np


def barycentric_to_r2(p1, p2, p3, z):
    # mapping (xhi1, xhi2, xhi3) to (x, y) given matrix of xhi-s Z
    return np.multiply.outer(p1, z[:, 0]) + np.multiply.outer(p2, z[:, 1]) + np.multiply.outer(p3, z[:, 2])


def get_area(p1, p2, p3):
    """
    Get the area of the triangle with vertices in p1, p2 and p3

    Parameters
    ----------
    p1 : np.array
        point p1.
    p2 : np.array
        point p2.
    p3 : np.array
        point p3.

    Returns
    -------
    float
        area of triangle.

    """
    det_jac = (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    return 0.5 * abs(det_jac)


def get_points_and_weights_quad_2D(nq):
    """
    Get Gauss quadrature points and weighs in 2D

    Parameters
    ----------
    nq : int
        scheme order.

    Raises
    ------
    ValueError
        if nq is not {1, 3, 4}.

    Returns
    -------
    z : np.array
        quadrature points.
    rho : np.array
        quadrature weights.

    """
    # Weights and gaussian quadrature points
    if nq == 1:
        z = np.ones((1, 3), dtype=float) / 3
        rho = np.ones(1, dtype=float)
    elif nq == 3:
        z = np.array([[0.5, 0.5, 0],
                      [0.5, 0, 0.5],
                      [0, 0.5, 0.5]], dtype=float)
        rho = np.ones(3, dtype=float) / 3
    elif nq == 4:
        z = np.array([[1 / 3, 1 / 3, 1 / 3],
                      [3 / 5, 1 / 5, 1 / 5],
                      [1 / 5, 3 / 5, 1 / 5],
                      [1 / 5, 1 / 5, 3 / 5]],
                     dtype=float)
        rho = np.array([-9 / 16,
                        25 / 48,
                        25 / 48,
                        25 / 48], dtype=float)
    else:
        raise ValueError("nq must in {1, 3, 4} for 2D quadrature")

    return z, rho


def quadrature2D(p1, p2, p3, g, nq):
    """
    Integral over the triangle with vertices in p1, p2 and p3

    Parameters
    ----------
    p1 : np.array
        point p1.
    p2 : np.array
        point p2.
    p3 : np.array
        point p3.
    g : function
        the function to integrate.
    nq : int
        scheme order.

    Returns
    -------
    float
        value of integral.

    """
    # Weights and gaussian quadrature points
    z, rho = get_points_and_weights_quad_2D(nq)

    # Calculating the Gaussian quadrature summation formula
    return get_area(p1, p2, p3) * np.sum(rho * g(*barycentric_to_r2(p1, p2, p3, z)))


def quadrature2D_vector(p1, p2, p3, g, nq):
    """
    Integrals over the triangle with vertices in p1, p2 and p3, assuming that g = [g1, g2], return int g1 and int g2

    Parameters
    ----------
    p1 : np.array
        point p1.
    p2 : np.array
        point p2.
    p3 : np.array
        point p3.
    g : function
        the vector function g = [g1, g2] to integrate.
    nq : int
        scheme order.

    Returns
    -------
    np.array
        values of the integrals' int g1 and int g2.

    """
    # Weights and gaussian quadrature points
    z, rho = get_points_and_weights_quad_2D(nq)

    # Calculating the Gaussian quadrature summation formula
    return get_area(p1, p2, p3) * np.sum(rho * g(*barycentric_to_r2(p1, p2, p3, z)), axis=1)
