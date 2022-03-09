# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from itertools import product

import numpy as np
from time import perf_counter
import scipy.sparse as sparse

from helpers import get_vec_from_range, get_lambda_mu
from old_scaling_of_rectangle.scalable_rectangle_class import ScalableRectangle
import default_constants


def get_parameter_vecs(m, mu_params, rec: ScalableRectangle,
                       e_young_range=None, nu_poisson_range=None, sample_mode="uniform"):
    parameter_vecs = []

    for mu in mu_params:
        if mu in rec.geo_mu_params:
            parameter_vecs.append(get_vec_from_range(rec.geo_mu_params_range[mu], m, sample_mode))
        else:
            # mu == "material_e_nu
            if e_young_range is None:
                e_young_range = default_constants.e_young_range
            else:
                e_young_range = e_young_range
            parameter_vecs.append(get_vec_from_range(e_young_range, m, sample_mode))

            if nu_poisson_range is None:
                nu_poisson_range = default_constants.nu_poisson_range
            else:
                nu_poisson_range = nu_poisson_range
            parameter_vecs.append(get_vec_from_range(nu_poisson_range, m, sample_mode))
    return parameter_vecs


def get_param_matrix(parameter_vecs):
    return np.array(list(product(*parameter_vecs)))


def compute_m_mat(m, mu_params, rec: ScalableRectangle, parameter_vecs, do_exrta=True):
    m4 = m ** len(parameter_vecs)
    if "material_e_nu" in mu_params:
        c_extra = 0
        if do_exrta:
            c_extra = 3
        m_mat = np.zeros((m4, 2 * len(mu_params) + c_extra), dtype=float)
        for i, params in enumerate(product(*parameter_vecs)):
            mu, lam = get_lambda_mu(*params[-2:])
            mls_funcs = rec.mls_funcs(*params[:-2])  # use this for now, more general needed later in the other examples...
            m_mat[i, 0:3] = 2 * mu * mls_funcs
            m_mat[i, 3:6] = lam * mls_funcs
            if do_exrta:
                # extra + 3
                m_mat[i, 6] = 1  # +3
                m_mat[i, 7] = lam * 2 * mu
                m_mat[i, 8] = params[0] * params[1]
    else:
        m_mat = np.zeros((m4, len(mu_params) + 1), dtype=float)
        # m_mat[:, 0] = 1 # 6->7 above
        for i, params in enumerate(product(*parameter_vecs)):
            m_mat[i, 0:3] = rec.mls_funcs(*params)
    return m_mat


def compute_b_mats_array(m, n, parameter_vecs, compute_a):
    n2d = n * n * 2
    m4 = m ** 4
    b_mats = np.zeros((m4, n2d, n2d), dtype=float)
    for i, params in enumerate(product(*parameter_vecs)):
        b_mats[i] = compute_a(*params).A
    return b_mats


def load_b_mats_array(file_directory):
    # load b_mats similar to as they are computed above
    ...


def load_a_mat(file_directory):
    # load one a_mat in b_mats similar to as they are computed above in the for loop,
    # should be done via class...
    ...


def matrix_least_squares(m, rec: ScalableRectangle, mu_params=None, mls_mode=None, compute_a_str=None,
                         e_young_range=None, nu_poisson_range=None, sample_mode="uniform"):
    # It should be possible to get the a-matrices (in b_mats) from saves here...
    if mls_mode is None:
        mls_mode = "sp"

    if mu_params is None:
        mu_params = rec.geo_mu_params.copy()

    if compute_a_str == "a1":
        compute_a = rec.compute_a1
    elif compute_a_str == "a2":
        compute_a = rec.compute_a2
    else:
        compute_a = rec.compute_a
        if "material_e_nu" not in mu_params:
            mu_params.append("material_e_nu")

    parameter_vecs = get_parameter_vecs(m, mu_params, rec, e_young_range, nu_poisson_range, sample_mode)

    s = perf_counter()
    m_mat = compute_m_mat(m, mu_params, rec, parameter_vecs)
    print("time m_mat:", perf_counter() - s)
    s = perf_counter()
    c_mat = np.linalg.solve(m_mat.T @ m_mat, m_mat.T)
    print("time c_mat:", perf_counter() - s)

    if mls_mode == "sp":
        # default sample_mode
        # use sp matrices and get the a-matrices (in b_mats) just in time.
        # the slowest method, but uses the least memory of the method.
        n, q_max = m_mat.shape
        n2d = rec.n * rec.n * 2
        ind = np.arange(n2d)
        c_mat = sparse.csr_matrix(c_mat)

        s = perf_counter()
        param_mat = get_param_matrix(parameter_vecs)
        print("time param_mat:", perf_counter() - s)

        s = perf_counter()
        x_mats = sparse.dok_matrix((n2d * q_max, n2d))
        for k in range(n):
            x_mats += sparse.kron(c_mat[:, k], compute_a(*param_mat[k]))
        print("time x_mats:", perf_counter() - s)
        x_mats = np.array([x_mats[np.ix_(ind + n2d * q, ind)] for q in range(q_max)])

    elif mls_mode == "array":
        # use arrays and get the all the a-matrices (in b_mats) before computing.
        # the fastest, but uses the most memory
        s = perf_counter()
        b_mats = compute_b_mats_array(m, rec.n, parameter_vecs, compute_a)
        print("time b_mats:", perf_counter() - s)

        s = perf_counter()
        x_mats = np.einsum("qk,kij -> qij", c_mat, b_mats)
        print("time x_mats2:", perf_counter() - s)
    else:
        error_text = "Mode is not implemented. Implemented modes: " + str(["sp", "array"])
        raise NotImplementedError(error_text)

    return x_mats


if __name__ == "__main__":
    m = np.array([[1, 100, 1_000],
                  [1, 2, 3]], dtype=float)
    c = np.linalg.solve(m.T @ m, m.T)
    """a = np.arange(2 * 2 * 2).reshape((2 * 2, 2))
    print(a)
    d = a.reshape(2, 2 * 2)
    print(d)
    print("dok_matrix")
    a = np.arange(2 * 2 * 2).reshape((2 * 2, 2))
    a = sp.dok_matrix(a)
    print(a.A)
    d = a.reshape(2, 2 * 2)
    print(d.A)
    print(c)
    u = c @ d
    print(u)
    print(type(u))
    v = u.reshape((3 * 2, 2))
    print(v)"""
    print("c", c)
    d = np.arange(c.size).reshape(c.shape) + 1
    print("d", d)
    a1 = np.arange(2 * 2).reshape((2, 2)) + 1
    a1 = sparse.dok_matrix(a1)
    a2 = np.arange(2 * 2).reshape((2, 2)) + 5
    a2 = sparse.dok_matrix(a2)
    a_list = [a1, a2]
    a0 = sparse.vstack(a_list)
    print("a0", a0.A)
    u = (d @ a0.reshape(2, 2 * 2)).reshape((3 * 2, 2))
    # print(x)
    b = sparse.dok_matrix((3 * 2, 2))
    for q in range(2):
        b += sparse.kron(d[:, q], a_list[q].T).T.A
        print(sparse.kron(d[:, q], a_list[q].T).T.A)

    print("u", u)
    print("b", b)
