# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
based on Specialization-Project-fall-2021
"""
from itertools import product

import numpy as np
from scipy.linalg import eigh, fractional_matrix_power

from default_constants import nu_poisson_range, e_young_range
from helpers import get_vec_from_range
# from scaling_of_rectangle.scalable_rectangle_class import ScalableRectangle


def make_solution_matrix(ns, rec_scale_vec, e_young_vec, nu_poisson_vec, rec):
    """

    Parameters
    ----------
    ns : int
        number of snapshots.
    e_young_vec : TYPE
        array of young's modules.
    nu_poisson_vec : np.array
        array of poisson ratios.
    rec :
        the solver.
    Raises
    ------
    SolutionMatrixIsZeroCanNotComputePODError
        If all values is the snapshot matrix s_mat are zero.
    Returns
    -------
    s_mat : np.array
        snapshot matrix.
    """
    s_mat = np.zeros((rec.n_free, ns))
    # solve system for all combinations of (e_young, nu_poisson)
    for i, (lx, ly, e_young, nu_poisson) in enumerate(product(rec_scale_vec, rec_scale_vec, e_young_vec, nu_poisson_vec)):
        rec.hfsolve(lx, ly, e_young, nu_poisson)
        s_mat[:, i] = rec.uh_free
    if (s_mat == 0).all():
        error_text = "Solution matrix is zero, can not compute POD for building a reduced model. " \
                     + "The most likely cause is f_func=0, dirichlet_bc_func=0 and neumann_bc_func=0, " \
                     + "where two last may be None."
        raise ValueError(error_text)
    return s_mat


def pod_with_energy_norm(m, rec, mode):
    """
    Proper orthogonal decomposition with respect to the energy norm
    Parameters
    ----------
    m: int
        grid parameter
    rec :
        the solver.
    Returns
    -------
    None.
    """
    rec_scale_vec = get_vec_from_range(rec.rec_scale_range, m, mode)
    e_young_vec = get_vec_from_range(e_young_range, m, mode)
    nu_poisson_vec = get_vec_from_range(nu_poisson_range, m, mode)

    l_mean = np.mean(rec.rec_scale_range)
    e_mean = np.mean(e_young_range)
    nu_mean = np.mean(nu_poisson_range)

    rec.ns_rom = m ** 4

    rec.s_mat = make_solution_matrix(rec.ns_rom, rec_scale_vec, e_young_vec, nu_poisson_vec, rec)
    a_free = rec.compute_a_free(l_mean, l_mean, e_mean, nu_mean)
    if rec.ns_rom <= rec.n_free:
        # build correlation matrix
        corr_mat = rec.s_mat.T @ a_free @ rec.s_mat
        # find the eigenvalues and eigenvectors of it
        sigma2_vec, z_mat = eigh(corr_mat)
        # reverse arrays because they are in ascending order
        rec.sigma2_vec = sigma2_vec[::-1]
        rec.z_mat_funcs = z_mat[:, ::-1]
    else:
        rec.x05 = fractional_matrix_power(a_free.A, 0.5)
        # build correlation matrix
        corr_mat = rec.x05 @ rec.s_mat @ rec.s_mat.T @ rec.x05
        # find the eigenvalues and eigenvectors of it
        sigma2_vec, z_mat = eigh(corr_mat)
        # reverse arrays because they are in ascending order
        rec.sigma2_vec = sigma2_vec[::-1]
        rec.z_mat_funcs = z_mat[:, ::-1]
    # compute n_rom from relative information content
    i_n = np.cumsum(rec.sigma2_vec) / np.sum(rec.sigma2_vec)
    rec.n_rom = np.min(np.argwhere(i_n >= 1 - rec.eps_pod ** 2)) + 1


def compute_v(n_rom, n_free, rec):
    """
    Compute the matrix V
    Parameters
    ----------
    n_rom : int
        our chosen "reduced-order degrees of freedom" ("n_rom"),
        can be set to different from n_rom-true.
    n_free : int
        the high-fidelity degrees of freedom.
    rec :
        reduced-order data.
    Returns
    -------
    None.
    """
    if rec.ns_rom <= n_free:
        rec.v = rec.s_mat @ rec.z_mat_funcs[:, :n_rom] / np.sqrt(rec.sigma2_vec[:n_rom])
    else:
        rec.v = np.linalg.solve(rec.x05, rec.z_mat_funcs[:, :n_rom])
