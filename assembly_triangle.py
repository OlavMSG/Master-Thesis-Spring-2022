# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
based on Specialization-Project-fall-2021
"""

import numpy as np

from gauss_quadrature import quadrature2D_tri
from helpers import inv_index_map, expand_index, VectorizedFunction2D


def get_basis_coef(p_vec):
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


def phi(x, y, ck, i):
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


def assemble_f_local(ck, f_func, p1, p2, p3):
    """
    Assemble the local contribution to the f_load_lv for the triangle element

    Parameters
    ----------
    ck : np.array
        basis function coef. matrix.
    f_func : function, VectorizedFunction2D
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
            return f_func(x, y)[:, di] * phi(x, y, ck, i)

        f_local[ki] = quadrature2D_tri(p1, p2, p3, 4, f_phi)
    return f_local


def assemble_f(n, p, tri, f_func, f_func_is_not_zero):
    """
    Assemble the load vector f_load_lv for triangle elements

    Parameters
    ----------
    n : int
        number of node along one axis.
    p : np.array
        list of points.
    tri : np.array
        triangulation of the points in p.
    f_func : function, VectorizedFunction2D
        load function.
    f_func_is_not_zero : bool
        True if f_func does not return (0,0) for all (x,y)

    Returns
    -------
    f_load_lv : np.array
        load vector for the linear form.

    """
    n2d = n * n * 2
    # load vector
    f_load_lv = np.zeros(n2d, dtype=float)
    if f_func_is_not_zero:
        for nk in tri:
            # nk : node-numbers for the k'th triangle
            # the points of the triangle
            # p1 = p[nk[0], :]
            # p2 = p[nk[1], :]
            # p3 = p[nk[2], :]
            # using indexmap k = 2 * i + d, d=0 for x, 1 for y, i is the node number
            # and basis functions coef. or Jacobin inverse
            ck = get_basis_coef(p[nk, :])
            # expand the index
            expanded_nk = expand_index(nk)
            # add local contributions
            f_load_lv[expanded_nk] += assemble_f_local(ck, f_func, *p[nk, :])
    return f_load_lv
