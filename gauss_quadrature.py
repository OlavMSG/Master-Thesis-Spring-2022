# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
from Specialization-Project-fall-2021
"""

import numpy as np
from scipy.special import roots_legendre
from itertools import product


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


def barycentric_to_r2(p1, p2, p3, z):
    # mapping (xhi1, xhi2, xhi3) to (x, y) given matrix of xhi-s Z
    return np.multiply.outer(p1, z[:, 0]) + np.multiply.outer(p2, z[:, 1]) + np.multiply.outer(p3, z[:, 2])


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


def get_points_and_weights_quad_2D_tri(nq):
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


def quadrature2D_tri(p1, p2, p3, nq, g):
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
    z, rho = get_points_and_weights_quad_2D_tri(nq)

    # Calculating the Gaussian quadrature summation formula
    return get_area_triangle(p1, p2, p3) * np.sum(rho * g(*barycentric_to_r2(p1, p2, p3, z)))


def minus_one_one_to_r2(p1, p2, p3, p4, z):
    # mapping (xhi1, xhi2) to (x, y) given matrix of xhi-s Z
    return np.multiply.outer(p1, np.ones_like(z[:, 0])) \
           + 0.25 * (np.multiply.outer(p2 - p1, (1 + z[:, 0]) * (1 - z[:, 1]))
                     + np.multiply.outer(p3 - p1, (1 + z[:, 0]) * (1 + z[:, 1]))
                     + np.multiply.outer(p4 - p1, (1 - z[:, 0]) * (1 + z[:, 1])))


def det_jac_minus_one_one_to_r2(p1, p2, p3, p4, z):
    jac0 = ((p1 - p2 + p3 - p4) * z - p1 + p2 + p3 - p4).T
    jac1 = (p1 * (z - 1) + z * (-p2 + p3 - p4) - p2 + p3 + p4).T
    return 0.0625 * (jac0[0, :] * jac1[1, :] - jac1[0, :] * jac0[1, :])


def get_points_and_weights_quad_2D_quad(nq_x, nq_y):
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
    z : np.array
        quadrature points.
    rho : np.array
        quadrature weights.

    """
    z_x, rho_x = roots_legendre(nq_x)
    z_y, rho_y = roots_legendre(nq_y)

    z = np.array(list(product(z_x, z_y)))
    rho = np.kron(rho_x, rho_y)
    return z, rho


def quadrature2D_quad(p1, p2, p3, p4, g, nq_x, nq_y=None):
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
    p4 : np.array
        point p4.
    g : function
        the function to integrate.
    nq_x : int
        scheme order in x.
    nq_y : int
        scheme order in y, equal to nq_x if None. Default None.

    Returns
    -------
    float
        value of integral.

    """
    if nq_y is None:
        nq_y = nq_x
    # Weights and gaussian quadrature points
    z, rho = get_points_and_weights_quad_2D_quad(nq_x, nq_y)

    # Calculating the Gaussian quadrature summation formula
    return np.sum(rho * g(*minus_one_one_to_r2(p1, p2, p3, p4, z)) * det_jac_minus_one_one_to_r2(p1, p2, p3, p4, z))


def get_area_quadrilateral(p1, p2, p3, p4):
    """
    Get the area of the quadrilateral with vertices in p1, p2, p3 and p4

    Parameters
    ----------
    p1 : np.array
        point p1.
    p2 : np.array
        point p2.
    p3 : np.array
        point p3.
    p4 : np.array
        point p4.

    Returns
    -------
    float
        area of quadrilateral.

    """
    return 0.5 * abs((p1[0] - p3[0]) * (p2[1] - p4[1]) - (p2[0] - p4[0]) * (p1[1] - p3[0]))


if __name__ == "__main__":
    p = np.array([[-2, -2],
                  [2, -2],
                  [2, 2]])
    area = get_area_triangle(*p)
    print(area)
    z = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    xy = barycentric_to_r2(*p, z)
    print(xy)

    p = np.array([[-1, -3],
                  [2, -3],
                  [2, 2],
                  [-1, 2]])
    area = get_area_quadrilateral(*p)
    print(area)
    z = np.array([[-1, -1],
                  [1, -1],
                  [1, 1],
                  [-1, 1]])
    xy = minus_one_one_to_r2(*p, z)
    print(xy)

    z, rho = get_points_and_weights_quad_2D_quad(2, 3)
    print(z)
    print(rho)


    def func1(x, y):
        return (2 - x) * (2 - y)


    true_a = 56.25
    a = quadrature2D_quad(*p, func1, 2)
    print(a, true_a, abs(true_a - a))

    print(det_jac_minus_one_one_to_r2(*p, z))
