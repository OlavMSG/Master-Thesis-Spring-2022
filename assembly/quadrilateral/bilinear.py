# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
based on Specialization-Project-fall-2021
"""

import numpy as np
import scipy.sparse as sparse

from assembly.quadrilateral.gauss_quadrature import quadrature2D, quadrature2D_vector
from helpers import expand_index, index_map

from assembly.neumann.linear import assemble_f_neumann


def get_basis_coef(p_vec):
    """
    Calculate the basis function coef. matrix. for quadrilateral element

    Parameters
    ----------
    p_vec : np.array
        vertexes of the quadrilateral element.

    Returns
    -------
    np.array
        basis function coef. matrix.

    """
    # row_k: [1, x_k, y_k, x_k * y_k]
    mk = np.column_stack((np.ones(4), p_vec[:, 0], p_vec[:, 1], p_vec[:, 0] * p_vec[:, 1]))
    return np.linalg.inv(mk)  # here faster than solving Mk @ Ck = I_4


def phi(x, y, ck, i):
    """
    The linear basis functions on a quadrilateral

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


def ddx_phi(x, y, ck, i):
    """
    The linear basis functions on a quadrilateral

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


def ddy_phi(x, y, ck, i):
    """
    The linear basis functions on a quadrilateral

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


def assemble_ints_local(ck, z_mat_funcs, geo_params, p_mat):
    """
    Assemble the local contributions to the 6 integrals on an element.

    Parameters
    ----------
    ck : np.array
        basis function coef. matrix.
    z_mat_funcs: np.ndarray
        array of functions for z matrix.
    geo_params: np.array
        geometry parameters
    p_mat : np.array
        vertexes of the quadrilateral element.

    Returns
    -------
    int1_local : np.array
        local contribution to matrix int1.
    int2_local : np.array
        local contribution to matrix int2.
    int3_local : np.array
        local contribution to matrix int3.
    int4_local : np.array
        local contribution to matrix int4.
    int5_local : np.array
        local contribution to matrix int5.

    """
    int1_local = np.zeros((8, 8), dtype=float)
    int2_local = np.zeros((8, 8), dtype=float)
    int3_local = np.zeros((8, 8), dtype=float)
    int4_local = np.zeros((8, 8), dtype=float)
    int5_local = np.zeros((8, 8), dtype=float)
    nq = 2

    # since the derivatives are non-constant we need to do the all integrals in the loop
    # matrices are symmetric by construction, so only compute on part.
    d = np.array([0, 1])
    for i in range(4):
        ki0, ki1 = index_map(i, d)  # di = 0, di = 1
        for j in range(i + 1):
            kj0, kj1 = index_map(j, d)  # dj = 0, dj = 1

            # [u_i1*v_j1, u_i1*v_j2, u_i2*v_j1, u_i2*v_j2]

            def cij_func0(x, y):
                return ddx_phi(x, y, ck, i) * ddx_phi(x, y, ck, j)

            def cij_func1(x, y):
                return ddx_phi(x, y, ck, i) * ddy_phi(x, y, ck, j)

            def cij_func2(x, y):
                return ddy_phi(x, y, ck, i) * ddx_phi(x, y, ck, j)

            def cij_func3(x, y):
                return ddy_phi(x, y, ck, i) * ddy_phi(x, y, ck, j)

            # construct local ints

            # u 1-comp is nonzero, di = 0
            # v 1-comp is nonzero, dj = 0
            # [u_11*v_11, u_11*v_12, u_12*v_11, u_12*v_12]
            def int1_00_func(x, y):
                return cij_func0(x, y) * z_mat_funcs[0, 0](x, y) \
                       + cij_func1(x, y) * z_mat_funcs[1, 0](x, y) \
                       + cij_func2(x, y) * z_mat_funcs[2, 0](x, y) \
                       + cij_func3(x, y) * z_mat_funcs[3, 0](x, y)

            def int2_00_func(x, y):
                return cij_func0(x, y) * z_mat_funcs[0, 3](x, y) \
                       + cij_func1(x, y) * z_mat_funcs[1, 3](x, y) \
                       + cij_func2(x, y) * z_mat_funcs[2, 3](x, y) \
                       + cij_func3(x, y) * z_mat_funcs[3, 3](x, y)

            int1_00 = quadrature2D(*p_mat, int1_00_func, nq)
            int2_00 = quadrature2D(*p_mat, int2_00_func, nq)

            int1_local[ki0, kj0] = int1_00
            int2_local[ki0, kj0] = int2_00
            if ki0 != kj0:
                int1_local[kj0, ki0] = int1_00
                int2_local[kj0, ki0] = int2_00

            # u 1-comp is nonzero, di = 0
            # v 2-comp is nonzero, dj = 1
            # [u_11*v_21, u_11*v_22, u_12*v_21, u_12*v_22]
            def int3_01_func(x, y):
                return cij_func0(x, y) * z_mat_funcs[0, 2](x, y) \
                       + cij_func3(x, y) * z_mat_funcs[3, 2](x, y)

            def int4_01_func(x, y):
                return cij_func1(x, y) * z_mat_funcs[1, 2](x, y) \
                       + cij_func2(x, y) * z_mat_funcs[1, 1](x, y)

            def int5_01_func(x, y):
                return cij_func1(x, y) * z_mat_funcs[1, 1](x, y) \
                       + cij_func2(x, y) * z_mat_funcs[1, 2](x, y)

            int3_01 = quadrature2D(*p_mat, int3_01_func, nq)
            int4_01 = quadrature2D(*p_mat, int4_01_func, nq)
            int5_01 = quadrature2D(*p_mat, int5_01_func, nq)

            int3_local[ki0, kj1] = int3_01
            int4_local[ki0, kj1] = int4_01
            int5_local[ki0, kj1] = int5_01
            if ki0 != kj1:
                int3_local[kj1, ki0] = int3_01
                int4_local[kj1, ki0] = int4_01
                int5_local[kj1, ki0] = int5_01

            # u 2-comp is nonzero, di = 1
            # v 1-comp is nonzero, dj = 0
            # [u_21*v_11, u_21*v_12, u_22*v_11, u_22*v_12]
            def int3_10_func(x, y):
                return cij_func0(x, y) * z_mat_funcs[0, 1](x, y) \
                       + cij_func3(x, y) * z_mat_funcs[3, 1](x, y)

            int3_10 = quadrature2D(*p_mat, int3_10_func, nq)
            # int4_10 = int5_01
            # int5_10 = int4_01

            int3_local[ki1, kj0] = int3_10
            int4_local[ki1, kj0] = int5_01
            int5_local[ki1, kj0] = int4_01
            if ki1 != kj0:
                int3_local[kj0, ki1] = int3_10
                int4_local[kj0, ki1] = int5_01
                int5_local[kj0, ki1] = int4_01

            # u 2-comp is nonzero, di = 1
            # v 2-comp is nonzero, dj = 1
            # [u_21*v_21, u_21*v_22, u_22*v_21, u_22*v_22]
            # int1_11 = int2_00
            # int2_11 = int1_00

            int1_local[ki1, kj1] = int2_00
            int2_local[ki1, kj1] = int1_00
            if ki1 != kj1:
                int1_local[kj1, ki1] = int2_00
                int2_local[kj1, ki1] = int1_00

    return int1_local, int2_local, int3_local, int4_local, int5_local


def assemble_f_local(ck, f_func, p_mat):
    """
    Assemble the local contribution to the f_load_lv for the quadrilateral
     element

    Parameters
    ----------
    ck : np.array
        basis function coef. matrix.
    f_func : function, VectorizedFunction2D
        load function.
    p_mat : np.array
        vertexes of the quadrilateral element.

    Returns
    -------
    np.array
        local contribution to f_load_lv.

    """
    f_local = np.zeros(8, dtype=float)
    nq = 4
    d = np.array([0, 1])
    for i in range(4):
        def f_phi(x, y):
            # f = [f0, f2]
            # phi_i0 = [phi_i, 0], phi_i1 = [0, phi_i]
            return f_func(x, y) * phi(x, y, ck, i)

        ki0, ki1 = index_map(i, d)
        f_local[[ki0, ki1]] = quadrature2D_vector(*p_mat, f_phi, nq)
    return f_local


def assemble_ints_and_f_load_lv(n, p, tri, z_mat_funcs, geo_params, f_func, f_func_is_not_zero):
    """
    Assemble the ints matrices and the body force load vector

    Parameters
    ----------
    n : int
        number of node along one axis.
    p : np.array
        list of points.
    tri : np.array
        triangulation of the points in p.
    z_mat_funcs: np.ndarray
        array of functions for z matrix.
    geo_params: np.array
        geometry parameters
    f_func : function, VectorizedFunction2D
        load function.
    f_func_is_not_zero : bool
        True if f_func does not return (0,0) for all (x,y)

    Returns
    -------
    ints: tuple
        tuple of sparse matrices of ints
    f_load_lv : np.array
        load vector for the linear form.
    """
    n2d = n * n * 2
    # Stiffness matrices
    # dok_matrix
    # Allows for efficient O(1) access of individual elements
    int1 = sparse.dok_matrix((n2d, n2d), dtype=float)
    int2 = sparse.dok_matrix((n2d, n2d), dtype=float)
    int3 = sparse.dok_matrix((n2d, n2d), dtype=float)
    int4 = sparse.dok_matrix((n2d, n2d), dtype=float)
    int5 = sparse.dok_matrix((n2d, n2d), dtype=float)
    # load vector
    f_load_lv = np.zeros(n2d, dtype=float)
    for nk in tri:
        # nk : node-numbers for the k'th triangle
        # the points of the triangle
        # p1 = p[nk[0], :]
        # p2 = p[nk[1], :]
        # p3 = p[nk[2], :]
        # using indexmap k = 2 * i + d, d=0 for x, 1 for y, i is the node number
        # calculate the area of the triangle
        # and basis functions coef. or Jacobin inverse
        ck = get_basis_coef(p[nk, :])
        # assemble local contributions
        ints = assemble_ints_local(ck, z_mat_funcs, geo_params, p[nk, :])
        # expand the index
        expanded_nk = expand_index(nk)
        index = np.ix_(expanded_nk, expanded_nk)
        # add local contributions
        int1[index] += ints[0]
        int2[index] += ints[1]
        int3[index] += ints[2]
        int4[index] += ints[4]
        int5[index] += ints[5]
        if f_func_is_not_zero:
            # load vector
            f_load_lv[expanded_nk] += assemble_f_local(ck, f_func, p[nk, :])
    return (int1, int2, int3, int4, int5), f_load_lv
