# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
based on Specialization-Project-fall-2021
"""

import numpy as np

from gauss_quadrature import quadrature2D_tri, line_integral_with_basis, quadrature2D_rec
from helpers import inv_index_map, expand_index


def get_basis_coef_tri(p_vec):
    """
    Calculate the basis function coef. matrix. for triangle element

    Parameters
    ----------
    p_vec : np.array
         vertexes of the triangle element.

    Returns
    -------
    np.array
        basis function coef. matrix.

    """
    # row_k: [1, x_k, y_k]
    mk = np.column_stack((np.ones(3), p_vec[:, 0], p_vec[:, 1]))
    return np.linalg.inv(mk)  # here faster than solving Mk @ Ck = I_3


def phi_tri(x, y, ck, i):
    """
    The linear basis functions on a triangle

    Parameters
    ----------
    x : np.array
        x-values.
    y : np.array
        y-values.
    ck : np.array
        basis function coef. matrix.
    i : int
        which basis function to use.

    Returns
    -------
    numpy.array
        basis function in the points (x,y).

    """
    # Ck = [[ck_1,  ck_2,  ck_3 ],  1  row index 0
    #       [ckx_1, ckx_2, ckx_3],  x  row index 1
    #       [cky_1, cky_2, cky_3]]  y  row index 2
    # col in : 0  ,   1  ,  2
    # phi1 = lambda x, y: [1, x, y] @ Ck[:, 0]
    # phi2 = lambda x, y: [1, x, y] @ Ck[:, 1]
    # phi3 = lambda x, y: [1, x, y] @ Ck[:, 2]
    return ck[0, i] + ck[1, i] * x + ck[2, i] * y


def assemble_f_local_tri(ck, f_func, p1, p2, p3):
    """
    Assemble the local contribution to the f_load_lv for the triangle element

    Parameters
    ----------
    ck : np.array
        basis function coef. matrix.
    f_func : function
        load function.
    p1 : np.array
        first vertex of the triangle element.
    p2 : np.array
        second vertex of the triangle element.
    p3 : np.array
        third vertex of the triangle element.

    Returns
    -------
    np.array
        local contribution to f_load_lv.

    """
    f_local = np.zeros(6, dtype=float)

    for ki in range(6):
        i, di = inv_index_map(ki)

        def f_phi(x, y):
            return f_func(x, y)[:, di] * phi_tri(x, y, ck, i)

        f_local[ki] = quadrature2D_tri(p1, p2, p3, 4, f_phi)
    return f_local


def get_basis_coef_rec(p_vec):
    """
    Calculate the basis function coef. matrix. for rectangle element

    Parameters
    ----------
    p_vec : np.array
        vertexes of the rectangle element.

    Returns
    -------
    np.array
        basis function coef. matrix.

    """
    # row_k: [1, x_k, y_k, x_k * y_k]
    mk = np.column_stack((np.ones(4), p_vec[:, 0], p_vec[:, 1], p_vec[:, 0] * p_vec[:, 1]))
    return np.linalg.inv(mk)  # here faster than solving Mk @ Ck = I_4


def phi_rec(x, y, ck, i):
    """
    The linear basis functions on a rectangle

    Parameters
    ----------
    x : np.array
        x-values.
    y : np.array
        y-values.
    ck : np.array
        basis function coef. matrix.
    i : int
        which basis function to use.

    Returns
    -------
    numpy.array
        basis function in the points (x,y).

    """
    # Ck = [[ck_1,  ck_2,  ck_3, ck_4 ],  1  row index 0
    #       [ckx_1, ckx_2, ckx_3, ckx_4],  x  row index 1
    #       [cky_1, cky_2, cky_3, cky_4]]  y  row index 2
    #       [ckxy_1, ckxy_2, ckxy_3, ckxy_4]]  y  row index 3
    # col in : 0  ,   1  ,    2,     3
    # phi1 = lambda x, y: [1, x, y, xy] @ Ck[:, 0]
    # phi2 = lambda x, y: [1, x, y, xy] @ Ck[:, 1]
    # phi3 = lambda x, y: [1, x, y, xy] @ Ck[:, 2]
    # phi3 = lambda x, y: [1, x, y, xy] @ Ck[:, 3]
    return ck[0, i] + ck[1, i] * x + ck[2, i] * y + ck[3, i] * x * y


def ddx_phi_rec(x, y, ck, i):
    """
    The linear basis functions on a rectangle

    Parameters
    ----------
    x : np.array
        x-values.
    y : np.array
        y-values.
    ck : np.array
        basis function coef. matrix.
    i : int
        which basis function to use.

    Returns
    -------
    numpy.array
        basis function in the points (x,y).

    """
    # Ck = [[ck_1,  ck_2,  ck_3, ck_4 ],  1  row index 0
    #       [ckx_1, ckx_2, ckx_3, ckx_4],  x  row index 1
    #       [cky_1, cky_2, cky_3, cky_4]]  y  row index 2
    #       [ckxy_1, ckxy_2, ckxy_3, ckxy_4]]  y  row index 3
    # col in : 0  ,   1  ,    2,     3
    # phi1 = lambda x, y: [1, x, y, xy] @ Ck[:, 0]
    # phi2 = lambda x, y: [1, x, y, xy] @ Ck[:, 1]
    # phi3 = lambda x, y: [1, x, y, xy] @ Ck[:, 2]
    # phi3 = lambda x, y: [1, x, y, xy] @ Ck[:, 3]
    return ck[1, i] + ck[3, i] * y


def ddy_phi_rec(x, y, ck, i):
    """
    The linear basis functions on a rectangle

    Parameters
    ----------
    x : np.array
        x-values.
    y : np.array
        y-values.
    ck : np.array
        basis function coef. matrix.
    i : int
        which basis function to use.

    Returns
    -------
    numpy.array
        basis function in the points (x,y).

    """
    # Ck = [[ck_1,  ck_2,  ck_3, ck_4 ],  1  row index 0
    #       [ckx_1, ckx_2, ckx_3, ckx_4],  x  row index 1
    #       [cky_1, cky_2, cky_3, cky_4]]  y  row index 2
    #       [ckxy_1, ckxy_2, ckxy_3, ckxy_4]]  y  row index 3
    # col in : 0  ,   1  ,    2,     3
    # phi1 = lambda x, y: [1, x, y, xy] @ Ck[:, 0]
    # phi2 = lambda x, y: [1, x, y, xy] @ Ck[:, 1]
    # phi3 = lambda x, y: [1, x, y, xy] @ Ck[:, 2]
    # phi3 = lambda x, y: [1, x, y, xy] @ Ck[:, 3]
    return ck[2, i] + ck[3, i] * x


def assemble_f_local_rec(ck, f_func, p1, p2, p3, p4):
    """
    Assemble the local contribution to the f_load_lv for the triangle element

    Parameters
    ----------
    ck : np.array
        basis function coef. matrix.
    f_func : function
        load function.
    p1 : np.array
        first vertex of the rectangle element.
    p2 : np.array
        second vertex of the rectangle element.
    p3 : np.array
        third vertex of the rectangle element.
    p4 : np.array
        forth vertex of the rectangle element.

    Returns
    -------
    np.array
        local contribution to f_load_lv.

    """
    f_local = np.zeros(8, dtype=float)

    for ki in range(8):
        i, di = inv_index_map(ki)

        def f_phi(x, y):
            return f_func(x, y)[:, di] * phi_rec(x, y, ck, i)

        f_local[ki] = quadrature2D_rec(p1, p2, p3, p4, f_phi, 4)
    return f_local


def assemble_f_neumann(n, p, neumann_edge, neumann_bc_func):
    """
    Assemble the neumann load vector

    Parameters
    ----------
    n : int
        number of node along one axis.
    p : np.array
        list of points.
    neumann_edge : np.array
        array of the edges of the triangulation.
    neumann_bc_func : function
        the neumann boundary condition function.

    Returns
    -------
    f_load_neumann : np.array
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
        f_load_neumann[expanded_ek] += line_integral_with_basis(*p[ek, :], 4, neumann_bc_func)
    return f_load_neumann
