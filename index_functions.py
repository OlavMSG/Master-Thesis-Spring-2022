# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""


def kron_delta(i, j):
    if i == j:
        return 1
    else:
        return 0


def i_hat(i, j):
    return j + 2 * kron_delta(1, i)


def j_hat(i, j):
    return i + 2 * kron_delta(1, j)


def k_hat(i):
    return 3 * kron_delta(1, i)


def l_hat(i):
    return kron_delta(0, i)


def n_hat(i):
    return 1 + kron_delta(0, i)


def m_hat(i):
    return 1 + kron_delta(1, i)


def h_hat(i):
    return 3 * kron_delta(0, i)
