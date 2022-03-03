# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
based on Specialization-Project-fall-2021
"""

import numpy as np
import scipy.sparse as sparse

from assembly.quadrilateral.bilinear import ddy_phi, ddx_phi, get_basis_coef
from assembly.quadrilateral.gauss_quadrature import quadrature2D
from helpers import expand_index, index_map


def assemble_ints_local(ck, p_vec):
    """
    Assemble the local contributions to the 6 integrals on an element.

    Parameters
    ----------
    ck : np.array
        basis function coef. matrix.
    p_vec : np.array
        vertexes of the rectangle element.

    Returns
    -------
    int11_local : np.array
        local contribution to matrix int11.
    int12_local : np.array
        local contribution to matrix int12.
    int21_local : np.array
        local contribution to matrix int21.
    int22_local : np.array
        local contribution to matrix int22.
    int4_local : np.array
        local contribution to matrix int4.
    int5_local : np.array
        local contribution to matrix int5.

    """
    int11_local = np.zeros((8, 8), dtype=float)
    int12_local = np.zeros((8, 8), dtype=float)
    int21_local = np.zeros((8, 8), dtype=float)
    int22_local = np.zeros((8, 8), dtype=float)
    int4_local = np.zeros((8, 8), dtype=float)
    int5_local = np.zeros((8, 8), dtype=float)

    # matrices are symmetric by construction, so only compute on part.
    for i in range(4):
        ki0 = index_map(i, 0)  # di = 0
        ki1 = index_map(i, 1)  # di = 1
        for j in range(i + 1):
            kj0 = index_map(j, 0)  # dj = 0
            kj1 = index_map(j, 1)  # dj = 1
            # [u_i1*v_j1, u_i1*v_j2, u_i2*v_j1, u_i2*v_j2]

            def cij_func0(x, y):
                return ddx_phi(x, y, ck, i) * ddx_phi(x, y, ck, j)

            def cij_func1(x, y):
                return ddx_phi(x, y, ck, i) * ddy_phi(x, y, ck, j)

            def cij_func2(x, y):
                return ddy_phi(x, y, ck, i) * ddx_phi(x, y, ck, j)

            def cij_func3(x, y):
                return ddy_phi(x, y, ck, i) * ddy_phi(x, y, ck, j)

            nq = 2
            cij0 = quadrature2D(*p_vec, cij_func0, nq)
            cij1 = quadrature2D(*p_vec, cij_func1, nq)
            cij2 = quadrature2D(*p_vec, cij_func2, nq)
            cij3 = quadrature2D(*p_vec, cij_func3, nq)

            # construct local ints

            # u 1-comp is nonzero, di = 0
            # v 1-comp is nonzero, dj = 0
            # [u_11*v_11, u_11*v_12, u_12*v_11, u_12*v_12]
            int11_local[ki0, kj0] = cij0
            int22_local[ki0, kj0] = cij3
            if ki0 != kj0:
                int11_local[kj0, ki0] = cij0
                int22_local[kj0, ki0] = cij3

            # u 1-comp is nonzero, di = 0
            # v 2-comp is nonzero, dj = 1
            # [u_11*v_21, u_11*v_22, u_12*v_21, u_12*v_22]
            int4_local[ki0, kj1] = cij2
            int5_local[ki0, kj1] = cij1
            if ki0 != kj1:
                int4_local[kj1, ki0] = cij2
                int5_local[kj1, ki0] = cij1

            if i != j:
                # u 2-comp is nonzero, di = 1
                # v 1-comp is nonzero, dj = 0
                # [u_21*v_11, u_21*v_12, u_22*v_11, u_22*v_12]
                int4_local[ki1, kj0] = cij1
                int5_local[ki1, kj0] = cij2
                if ki1 != kj0:
                    int4_local[kj0, ki1] = cij1
                    int5_local[kj0, ki1] = cij2

            # u 2-comp is nonzero, di = 1
            # v 2-comp is nonzero, dj = 1
            # [u_21*v_21, u_21*v_22, u_22*v_21, u_22*v_22]
            int12_local[ki1, kj1] = cij3
            int21_local[ki1, kj1] = cij0
            if ki1 != kj1:
                int12_local[kj1, ki1] = cij3
                int21_local[kj1, ki1] = cij0

    return int11_local, int12_local, int21_local, int22_local, int4_local, int5_local


def assemble_ints_quad(n, p, tri):
    """
    Assemble the matrices for ints

    Parameters
    ----------
    n : int
        number of node along one axis.
    p : np.array
        list of points.
    tri : np.array
        triangulation of the points in p.

    Returns
    -------
    int11 : sparse.dok_matrix
        matrix int11 for the bilinear form.
    int12 : sparse.dok_matrix
        matrix int12 for the bilinear form.
    int21 : sparse.dok_matrix
        matrix int21 for the bilinear form.
    int22 : sparse.dok_matrix
        matrix int22 for the bilinear form.
    int4 : sparse.dok_matrix
        matrix int4 for the bilinear form.
    int5 : sparse.dok_matrix
        matrix int5 for the bilinear form.
    """
    n2d = n * n * 2
    # Stiffness matrices
    int11 = sparse.dok_matrix((n2d, n2d), dtype=float)
    int12 = sparse.dok_matrix((n2d, n2d), dtype=float)
    int21 = sparse.dok_matrix((n2d, n2d), dtype=float)
    int22 = sparse.dok_matrix((n2d, n2d), dtype=float)
    int4 = sparse.dok_matrix((n2d, n2d), dtype=float)
    int5 = sparse.dok_matrix((n2d, n2d), dtype=float)
    # dok_matrix
    # Allows for efficient O(1) access of individual elements
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
        ints = assemble_ints_local(ck, p[nk, :])
        # expand the index
        expanded_nk = expand_index(nk)
        index = np.ix_(expanded_nk, expanded_nk)
        # add local contributions
        int11[index] += ints[0]
        int12[index] += ints[1]
        int21[index] += ints[2]
        int22[index] += ints[3]
        int4[index] += ints[4]
        int5[index] += ints[5]
    return int11, int12, int21, int22, int4, int5

