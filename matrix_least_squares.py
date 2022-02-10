# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np


def matrix_least_squares(m_mat, b_mats):
    # n, q_max = m_mat.shape
    # n2 = b_mats.shape[1]

    c_mat = np.linalg.solve(m_mat.T @ m_mat, m_mat.T)

    x_mats = np.einsum("qk,kij -> qij", c_mat, b_mats)

    """x_mats2 = np.zeros((q_max, n2, n2))
    for q in range(q_max):
        for k in range(n):
            x_mats2[q, :, :] += c_mat[q, k] * b_mats[k, :, :]

    print((np.abs(x_mats - x_mats2) < 1e-14).all(), "hei")"""

    return x_mats
