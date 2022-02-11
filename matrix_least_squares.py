# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from itertools import product

import numpy as np
from time import perf_counter
import scipy.sparse as sparse

from helpers import get_vec_from_range, get_lambda_mu
from scaling_of_rectangle.scalable_rectangle_class import ScalableRectangle
import default_constants


def compute_m_mat(m, rec: ScalableRectangle, rec_scale_vec, e_young_vec, nu_poisson_vec):
    m4 = m ** 4
    m_mat = np.zeros((m4, 6), dtype=float)
    # m_mat[:, 0] = 1 # 6->7 above
    for i, (lx, ly, e_young, nu_poisson) in \
            enumerate(product(rec_scale_vec, rec_scale_vec, e_young_vec, nu_poisson_vec)):
        mu, lam = get_lambda_mu(e_young, nu_poisson)
        mls_funcs = rec.mls_funcs(lx, ly)  # use this for now, more general needed later in the other examples...
        m_mat[i, 0:3] = 2 * mu * mls_funcs
        m_mat[i, 3:] = lam * mls_funcs
        # m_mat[i, 7] = lam * 2 * mu
        # m_mat[i, 8] = lx * ly
    return m_mat


def compute_b_mats(m, rec: ScalableRectangle, rec_scale_vec, e_young_vec, nu_poisson_vec):
    n2d = rec.n * rec.n * 2
    m4 = m ** 4
    # save memory by using sparse matrices.
    # b_mats = np.zeros((m4, n2d, n2d), dtype=float)
    b_mats = sparse.dok_matrix((m4 * n2d, n2d), dtype=float)
    ind = np.arange(n2d)
    for i, (lx, ly, e_young, nu_poisson) in \
            enumerate(product(rec_scale_vec, rec_scale_vec, e_young_vec, nu_poisson_vec)):
        rec.set_param(lx, ly)
        mu, lam = get_lambda_mu(e_young, nu_poisson)

        a1, a2 = rec.compute_a1_and_a2_from_ints()

        index = np.ix_(ind + n2d * i, ind)
        b_mats[index] = 2 * mu * a1 + lam * a2
    return b_mats


def load_b_mats(file_directory):
    ...


def get_param_matrix(rec_scale_vec, e_young_vec, nu_poisson_vec):
    return np.array(list(product(rec_scale_vec, rec_scale_vec, e_young_vec, nu_poisson_vec)))


def compute_a_mat(rec: ScalableRectangle, lx, ly, e_young, nu_poisson):
    return rec.compute_a(lx, ly, e_young, nu_poisson)


def load_a_mat(file_directory):
    ...


def matrix_least_squares(m: int, rec: ScalableRectangle, e_young_range=None, nu_poisson_range=None, mode="uniform",
                         mls_mode=None):
    # It should be possible to get the a-matrices (in b_mats) from saves here...
    if mls_mode is None:
        mls_mode = default_constants.mls_mode
    if e_young_range is None:
        e_young_range = default_constants.e_young_range
    else:
        e_young_range = e_young_range
    if nu_poisson_range is None:
        nu_poisson_range = default_constants.nu_poisson_range
    else:
        nu_poisson_range = nu_poisson_range

    rec_scale_vec = get_vec_from_range(rec.rec_scale_range, m, mode)
    e_young_vec = get_vec_from_range(e_young_range, m, mode)
    nu_poisson_vec = get_vec_from_range(nu_poisson_range, m, mode)

    s = perf_counter()
    m_mat = compute_m_mat(m, rec, rec_scale_vec, e_young_vec, nu_poisson_vec)
    print("time m_mat:", perf_counter() - s)
    s = perf_counter()
    c_mat = np.linalg.solve(m_mat.T @ m_mat, m_mat.T)
    print("time c_mat:", perf_counter() - s)

    n, q_max = m_mat.shape
    n2d = rec.n * rec.n * 2

    if mls_mode == "sparse jit":
        # default mode
        # use sparse matrices and get the a-matrices (in b_mats) just in time.
        # the slowest method, but uses the least memory of the methods.
        ind = np.arange(n2d)
        c_mat = sparse.csr_matrix(c_mat)

        s = perf_counter()
        param_mat = get_param_matrix(rec_scale_vec, e_young_vec, nu_poisson_vec)
        print("time param_mat:", perf_counter() - s)

        s = perf_counter()
        x_mats = sparse.dok_matrix((n2d * q_max, n2d))
        for k in range(n):
            x_mats += sparse.kron(c_mat[:, k], compute_a_mat(rec, *param_mat[k]))
        print("time x_mats:", perf_counter() - s)
        x_mats = np.array([x_mats[np.ix_(ind + n2d * q, ind)] for q in range(q_max)])
    elif mls_mode == "sparse":
        # use sparse matrices and get the all the a-matrices (in b_mats) before computing.
        # a bit faster than the default, but uses more memory
        ind = np.arange(n2d)
        c_mat = sparse.csr_matrix(c_mat)

        s = perf_counter()
        b_mats = compute_b_mats(m, rec, rec_scale_vec, e_young_vec, nu_poisson_vec)
        print("time b_mats:", perf_counter() - s)

        s = perf_counter()
        x_mats = (c_mat @ b_mats.reshape(n, n2d * n2d)).reshape(q_max * n2d, n2d).tocsr()
        x_mats = np.array([x_mats[np.ix_(ind + n2d * q, ind)] for q in range(q_max)])

    elif mls_mode == "array":
        # use arrays and get the all the a-matrices (in b_mats) before computing.
        # the fastest, but uses the most memory
        s = perf_counter()
        b_mats = compute_b_mats(m, rec, rec_scale_vec, e_young_vec, nu_poisson_vec).A.reshape((n, n2d, n2d))
        print("time b_mats:", perf_counter() - s)

        s = perf_counter()
        x_mats = np.einsum("qk,kij -> qij", c_mat, b_mats)
        print("time x_mats2:", perf_counter() - s)
    else:
        error_text = "Mode is not implemented. Implemented modes: " + str(default_constants.implemented_mls_modes)
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
    a = sparse.dok_matrix(a)
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
