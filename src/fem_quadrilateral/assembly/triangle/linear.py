# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
based on Specialization-Project-fall-2021
"""
from __future__ import annotations
from typing import Optional, Tuple, Callable, Union
import numpy as np
import scipy.sparse as sp

from .gauss_quadrature import quadrature2D, quadrature2D_vector
from ...helpers import expand_index, index_map

# should be accessible form this file, so import it
from ..neumann.linear import assemble_f_neumann

__all__ = [
    "assemble_f_neumann",
    "get_basis_coef",
    "phi",
    "nabla_grad",
    "assemble_a1_a2_local",
    "assemble_f_local",
    "assemble_a1_a2_and_f_body_force"
]


def get_basis_coef(p_vec: np.ndarray) -> np.ndarray:
    """
    Calculate the basis function coef. matrix. for triangle element

    Parameters
    ----------
    p_vec : np.ndarray
         vertexes of the triangle element.

    Returns
    -------
    np.ndarray
        basis function coef. matrix.

    """
    # row_k: [1, x_k, y_k]
    mk = np.column_stack((np.ones(3), p_vec[:, 0], p_vec[:, 1]))
    return np.linalg.inv(mk)  # here faster than solving Mk @ Ck = I_3


def phi(x: Union[np.ndarray, float], y: Union[np.ndarray, float], ck: np.ndarray, i: int) -> Union[np.ndarray, float]:
    """
    The triangle basis functions on a triangle

    Parameters
    ----------
    x : np.ndarray, float
        x-values.
    y : np.ndarray, float
        y-values.
    ck : np.ndarray
        basis function coef. matrix.
    i : int
        which basis function to use.

    Returns
    -------
    numpy.ndarray, float
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


def nabla_grad(ck: np.ndarray, i: int, d: int) -> np.ndarray:
    """

    Parameters
    ----------
    ck : np.ndarray
        basis function coef. matrix.
    i : int
        which basis function to use.
    d : int
        which dimension.
    Returns
    -------
    np.array
        reference gradient.
    """
    if d == 0:
        # case y-part equal 0 of basisfunc
        return np.array([[ck[1, i], 0.],
                         [ck[2, i], 0.]], dtype=float)
    else:
        # case x-part equal 0 of basisfunc
        return np.array([[0., ck[1, i]],
                         [0., ck[2, i]]], dtype=float)


def assemble_a1_a2_local(ck: np.ndarray, z_mat_funcs: np.ndarray, geo_params: np.ndarray,
                         p_mat: np.ndarray, nq: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble the local contributions to an element.

    Parameters
    ----------
    ck : np.ndarray
        basis function coef. matrix.
    z_mat_funcs: np.ndarray
        array of functions for z matrix.
    geo_params: np.ndarray
        geometry parameters
    p_mat : np.ndarray
        vertexes of the triangle element.
    nq : int
        quadrature scheme order.

    Returns
    -------
    a1_local : np.ndarray
        local contribution to matrix a1.
    a2_local : np.ndarray
        local contribution to matrix a2.

    """
    a1_local = np.zeros((6, 6), dtype=float)
    a2_local = np.zeros((6, 6), dtype=float)

    # since the derivatives are constant we only need the integral of all 16 functions in z_mat_funcs
    z_mat = np.zeros((4, 4), dtype=float)
    for i in range(4):
        for j in range(4):
            def z_mat_funcs_ij(x: np.ndarray, y: np.ndarray) -> np.ndarray:
                return z_mat_funcs[i, j](x, y, *geo_params)

            z_mat[i, j] = quadrature2D(*p_mat, g=z_mat_funcs_ij, nq=nq)

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

            a1_local[ki0, kj0] = int1_00 + 0.5 * int2_00
            a2_local[ki0, kj0] = int1_00
            if ki0 != kj0:
                a1_local[kj0, ki0] = int1_00 + 0.5 * int2_00
                a2_local[kj0, ki0] = int1_00

            # u 1-comp is nonzero, di = 0
            # v 2-comp is nonzero, dj = 1
            # [u_11*v_21, u_11*v_22, u_12*v_21, u_12*v_22]
            int3_01 = np.sum(cij[[0, 3]] * z_mat[[0, 3], 2])
            int4_01 = np.sum(cij[[1, 2]] * z_mat[1, [2, 1]])
            int5_01 = np.sum(cij[[1, 2]] * z_mat[1, [1, 2]])

            a1_local[ki0, kj1] = 0.5 * (int3_01 + int4_01)
            a2_local[ki0, kj1] = int3_01 + int5_01
            if ki0 != kj1:
                a1_local[kj1, ki0] = 0.5 * (int3_01 + int4_01)
                a2_local[kj1, ki0] = int3_01 + int5_01

            if i != j:
                # u 2-comp is nonzero, di = 1
                # v 1-comp is nonzero, dj = 0
                # [u_21*v_11, u_21*v_12, u_22*v_11, u_22*v_12]
                # int3_10 = np.sum(cij[[0, 3]] * z_mat[[0, 3], 1]) # int3_01
                # int4_10 = np.sum(cij[[1, 2]] * z_mat[2, [2, 1]])  # = int5_01
                # int5_10 = np.sum(cij[[1, 2]] * z_mat[2, [1, 2]])  # = int4_01

                a1_local[ki1, kj0] = 0.5 * (int3_01 + int5_01)
                a2_local[ki1, kj0] = int3_01 + int4_01
                if ki1 != kj0:
                    a1_local[kj0, ki1] = 0.5 * (int3_01 + int5_01)
                    a2_local[kj0, ki1] = int3_01 + int4_01

            # u 2-comp is nonzero, di = 1
            # v 2-comp is nonzero, dj = 1
            # [u_21*v_21, u_21*v_22, u_22*v_21, u_22*v_22]
            # int1_11 = np.sum(cij * z_mat[:, 3]) = int2_00
            # int2_11 = np.sum(cij * z_mat[:, 0]) = int1_00

            a1_local[ki1, kj1] = int2_00 + 0.5 * int1_00
            a2_local[ki1, kj1] = int2_00
            if ki1 != kj1:
                a1_local[kj1, ki1] = int2_00 + 0.5 * int1_00
                a2_local[kj1, ki1] = int2_00

    return a1_local, a2_local


def assemble_f_local(ck: np.ndarray, f_func: Callable, p_mat: np.ndarray, nq: int):
    """
    Assemble the local contribution to the f_load_lv for the triangle element

    Parameters
    ----------
    ck : np.ndarray
        basis function coef. matrix.
    f_func : Callable
        load function.
    p_mat : np.ndarray
        vertexes of the triangle element.
    nq : int
        quadrature scheme order.

    Returns
    -------
    np.ndarray
        local contribution to f_load_lv.

    """
    f_local = np.zeros(6, dtype=float)
    d = np.array([0, 1])
    for i in range(3):
        def f_phi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            # f = [f0, f2]
            # phi_i0 = [phi_i, 0], phi_i1 = [0, phi_i]
            return f_func(x, y) * phi(x, y, ck, i)

        ki0, ki1 = index_map(i, d)
        f_local[[ki0, ki1]] = quadrature2D_vector(*p_mat, g=f_phi, nq=nq)
    return f_local


def assemble_a1_a2_and_f_body_force(n: int, p: np.ndarray, tri: np.ndarray, z_mat_funcs: np.ndarray,
                                    geo_params: np.ndarray, f_func: Callable, f_func_is_not_zero: bool,
                                    nq: Optional[int] = 4) -> Tuple[sp.spmatrix, sp.spmatrix, np.ndarray]:
    """
    Assemble the a1 and a2 matrices and the body force load vector

    Parameters
    ----------
    n : int
        number of node along one axis.
    p : np.ndarray
        list of points.
    tri : np.ndarray
        triangulation of the points in p.
    z_mat_funcs: np.ndarray
        array of functions for z matrix.
    geo_params: np.ndarray
        geometry parameters
    f_func : Callable
        load function.
    f_func_is_not_zero : bool
        True if f_func does not return (0,0) for all (x,y)
    nq : int, optional
        triangle quadrature scheme order. The default is 4.

    Returns
    -------
    a1 : sp.spmatrix
        full matrix a1.
    a2 : sp.spmatrix
        full matrix a2.
    f_body_force : np.ndarray
        load vector for the triangle form.

    """
    n2d = n * n * 2
    # Stiffness matrices
    # dok_matrix
    # Allows for efficient O(1) access of individual elements
    a1 = sp.dok_matrix((n2d, n2d), dtype=float)
    a2 = sp.dok_matrix((n2d, n2d), dtype=float)
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
        a1_local, a2_local = assemble_a1_a2_local(ck, z_mat_funcs, geo_params, p[nk, :], nq)
        # expand the index
        expanded_nk = expand_index(nk)
        index = np.ix_(expanded_nk, expanded_nk)
        # add local contributions
        # matrices
        a1[index] += a1_local
        a2[index] += a2_local
        if f_func_is_not_zero:
            # load vector
            f_body_force[expanded_nk] += assemble_f_local(ck, f_func, p[nk, :], nq)
    return a1, a2, f_body_force
