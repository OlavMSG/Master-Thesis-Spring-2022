# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
based on Specialization-Project-fall-2021
"""

import numpy as np
import scipy.sparse as sp

from assembly.triangle.gauss_quadrature import quadrature2D, quadrature2D_vector
from helpers import expand_index, index_map

# should be accessible form this file, so import it
from assembly.neumann.linear import assemble_f_neumann


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


def assemble_ints_local(ck, z_mat_funcs, geo_params, p_mat, nq):
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
        vertexes of the triangle element.
    nq : int
        quadrature scheme order.

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
    int1_local = np.zeros((6, 6), dtype=float)
    int2_local = np.zeros((6, 6), dtype=float)
    int3_local = np.zeros((6, 6), dtype=float)
    int4_local = np.zeros((6, 6), dtype=float)
    int5_local = np.zeros((6, 6), dtype=float)

    # since the derivatives are constant we only need the integral of all 16 functions in z_mat_funcs
    z_mat = np.zeros((4, 4), dtype=float)
    for i in range(4):
        for j in range(4):
            def z_mat_funcs_ij(x, y):
                return z_mat_funcs[i, j](x, y, *geo_params)
            z_mat[i, j] = quadrature2D(*p_mat, z_mat_funcs_ij, nq)

    # matrices are symmetric by construction, so only compute on part.
    d = np.array([0, 1])
    for i in range(3):
        ki0, ki1 = index_map(i, d)  # di = 0, di = 1
        for j in range(i + 1):
            kj0, kj1 = index_map(j, d)  # dj = 0, dj = 1
            # [u_i1*v_j1, u_i1*v_j2, u_i2*v_j1, u_i2*v_j2]
            cij = np.kron(ck[1:3, i], ck[1:3, j])

            # construct local ints

            # u 1-comp is nonzero, di = 0
            # v 1-comp is nonzero, dj = 0
            # [u_11*v_11, u_11*v_12, u_12*v_11, u_12*v_12]
            int1_00 = np.sum(cij * z_mat[:, 0])
            int2_00 = np.sum(cij * z_mat[:, 3])

            int1_local[ki0, kj0] = int1_00
            int2_local[ki0, kj0] = int2_00
            if ki0 != kj0:
                int1_local[kj0, ki0] = int1_00
                int2_local[kj0, ki0] = int2_00

            # u 1-comp is nonzero, di = 0
            # v 2-comp is nonzero, dj = 1
            # [u_11*v_21, u_11*v_22, u_12*v_21, u_12*v_22]
            int3_01 = np.sum(cij[[0, 3]] * z_mat[[0, 3], 2])
            int4_01 = np.sum(cij[[1, 2]] * z_mat[1, [2, 1]])
            int5_01 = np.sum(cij[[1, 2]] * z_mat[1, [1, 2]])

            int3_local[ki0, kj1] = int3_01
            int4_local[ki0, kj1] = int4_01
            int5_local[ki0, kj1] = int5_01
            if ki0 != kj1:
                int3_local[kj1, ki0] = int3_01
                int4_local[kj1, ki0] = int4_01
                int5_local[kj1, ki0] = int5_01

            if i != j:
                # u 2-comp is nonzero, di = 1
                # v 1-comp is nonzero, dj = 0
                # [u_21*v_11, u_21*v_12, u_22*v_11, u_22*v_12]
                # int3_10 = np.sum(cij[[0, 3]] * z_mat[[0, 3], 1]) # int3_01
                # int4_10 = np.sum(cij[[1, 2]] * z_mat[2, [2, 1]])  # = int5_01
                # int5_10 = np.sum(cij[[1, 2]] * z_mat[2, [1, 2]])  # = int4_01

                int3_local[ki1, kj0] = int3_01
                int4_local[ki1, kj0] = int5_01
                int5_local[ki1, kj0] = int4_01
                if ki1 != kj0:
                    int3_local[kj0, ki1] = int3_01
                    int4_local[kj0, ki1] = int5_01
                    int5_local[kj0, ki1] = int4_01

            # u 2-comp is nonzero, di = 1
            # v 2-comp is nonzero, dj = 1
            # [u_21*v_21, u_21*v_22, u_22*v_21, u_22*v_22]
            # int1_11 = np.sum(cij * z_mat[:, 3]) = int2_00
            # int2_11 = np.sum(cij * z_mat[:, 0]) = int1_00

            int1_local[ki1, kj1] = int2_00
            int2_local[ki1, kj1] = int1_00
            if ki1 != kj1:
                int1_local[kj1, ki1] = int2_00
                int2_local[kj1, ki1] = int1_00
    # print((check_mat - 1 == 0).all())
    return int1_local, int2_local, int3_local, int4_local, int5_local


def assemble_f_local(ck, f_func, p_mat, nq):
    """
    Assemble the local contribution to the f_load_lv for the triangle element

    Parameters
    ----------
    ck : np.array
        basis function coef. matrix.
    f_func : function, VectorizedFunction2D
        load function.
    p_mat : np.array
        vertexes of the triangle element.
    nq : int
        quadrature scheme order.

    Returns
    -------
    np.array
        local contribution to f_load_lv.

    """
    f_local = np.zeros(6, dtype=float)
    d = np.array([0, 1])
    for i in range(3):
        def f_phi(x, y):
            # f = [f0, f2]
            # phi_i0 = [phi_i, 0], phi_i1 = [0, phi_i]
            return f_func(x, y) * phi(x, y, ck, i)

        f_local[index_map(i, d)] = quadrature2D_vector(*p_mat, f_phi, nq)
    return f_local


def assemble_ints_and_f_body_force(n, p, tri, z_mat_funcs, geo_params, f_func, f_func_is_not_zero, nq=4):
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
    nq : int, optional
        triangle quadrature scheme order. The default is 4.

    Returns
    -------
    ints: tuple
        tuple of sp matrices of ints
    f_body_force : np.array
        load vector for the linear form.

    """
    n2d = n * n * 2
    # Stiffness matrices
    # dok_matrix
    # Allows for efficient O(1) access of individual elements
    int1 = sp.dok_matrix((n2d, n2d), dtype=float)
    int2 = sp.dok_matrix((n2d, n2d), dtype=float)
    int3 = sp.dok_matrix((n2d, n2d), dtype=float)
    int4 = sp.dok_matrix((n2d, n2d), dtype=float)
    int5 = sp.dok_matrix((n2d, n2d), dtype=float)
    # load vector
    f_body_force = np.zeros(n2d, dtype=float)

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
        ints_local = assemble_ints_local(ck, z_mat_funcs, geo_params, p[nk, :], nq)
        # expand the index
        expanded_nk = expand_index(nk)
        index = np.ix_(expanded_nk, expanded_nk)
        # add local contributions
        # matrices
        int1[index] += ints_local[0]
        int2[index] += ints_local[1]
        int3[index] += ints_local[2]
        int4[index] += ints_local[3]
        int5[index] += ints_local[4]
        if f_func_is_not_zero:
            # load vector
            f_body_force[expanded_nk] += assemble_f_local(ck, f_func, p[nk, :], nq)
    return (int1, int2, int3, int4, int5), f_body_force
