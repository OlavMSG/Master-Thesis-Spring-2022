# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
from Specialization-Project-fall-2021
"""

import numpy as np
from scipy.special import roots_legendre


def line_integral_with_basis(a, b, nq, g):
    """
    Line integral with local basis functions on line from a to b

    Parameters
    ----------
    a : np.array
        point a.
    b : np.array
        point b.
    nq : int
            scheme order.
    g : function
        the function to integrate (times basis functions).

    Returns
    -------
    line_ints : np.array
        array of line integrals.

    """
    z, rho = roots_legendre(nq)
    # Weights and gaussian quadrature points
    # if nq == 1:
    #    z = np.zeros(1, dtype=float)
    #    rho = np.ones(1, dtype=float) * 2
    # if nq == 2:
    #    c = np.sqrt(1 / 3)
    #    z = np.array([-c, c], dtype=float)
    #    rho = np.ones(2, dtype=float)
    # if nq == 3:
    #     c = np.sqrt(3 / 5)
    #    z = np.array([-c, 0, c], dtype=float)
    #    rho = np.array([5, 8, 5], dtype=float) / 9
    # if nq == 4:
    #    c1 = np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7)
    #    c2 = np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7)
    #    z = np.array([-c1, -c2, c2, c1], dtype=float)
    #    k1 = 18 - np.sqrt(30)
    #    k2 = 18 + np.sqrt(30)
    #    rho = np.array([k1, k2, k2, k1], dtype=float) / 36

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


def quadrature2D(p1, p2, p3, nq, g):
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
    nq : int
        scheme order.
    g : function
        the function to integrate.

    Returns
    -------
    float
        value of integral.

    """
    # Weights and gaussian quadrature points
    z, rho = get_points_and_weights_quad_2D(nq)
    # calculate the area of the triangle
    area = get_area_triangle(p1, p2, p3)

    # Calculating the physical points
    # mapping (xhi1, xhi2, xhi3) to (x, y) given matrix of xhi-s Z
    xy = np.multiply.outer(p1, z[:, 0]) + np.multiply.outer(p2, z[:, 1]) + np.multiply.outer(p3, z[:, 2])
    # Calculating the Gaussian quadrature summation formula
    return area * np.sum(rho * g(*xy))


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


def get_area_triangle(p1, p2, p3):
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
