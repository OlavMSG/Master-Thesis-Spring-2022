# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from time import perf_counter

import numpy as np

from scaling_of_rectangle.scalable_rectangle_class import ScalableRectangle
from matrix_least_squares import matrix_least_squares

def main():
    n = 3
    m = 3
    # lx, ly = 3, 1
    rec = ScalableRectangle(n, "bq")
    # rec.set_param(lx, ly)
    e_young, nu_poisson = 2.1e5, 0.3
    tol = 1e-14

    s = perf_counter()
    rec.assemble_ints()
    # rec.assemble_f()
    print("time assemble:", perf_counter() - s)

    s = perf_counter()
    m_mat, b_mat = rec.compute_matrix_least_squares_m_and_b(m)
    print("time m_mat b_mat:", perf_counter() - s)
    # a1_full, a2_full = rec.compute_a1_and_a2_from_ints()
    # a_full = compute_a(e_young, nu_poisson, a1_full, a2_full)
    s = perf_counter()
    x_mats = matrix_least_squares(m_mat, b_mat)
    print("time matrix least squares:", perf_counter() - s)
    print("checking np.max(np.abs(...))")
    print(x_mats.shape)
    rec.set_param(1, 1)

    # a = rec.compute_a1_and_a2_from_ints()

    # print(np.max(np.abs(x_mats[0])))
    # print(np.max(np.abs(a[0] - x_mats[0])) < 1e-14)
    # print(np.max(np.abs(a[1] - x_mats[1])) < 1e-14)
    print("mu")
    print(np.max(np.abs(rec.ints[0] + 0.5 * rec.ints[2] - x_mats[0])))
    print(np.max(np.abs(rec.ints[1] + 0.5 * rec.ints[3] - x_mats[1])))
    print(np.max(np.abs(0.5 * rec.ints[4] - x_mats[2])))
    print("lambda")
    print(np.max(np.abs(rec.ints[0] - x_mats[3])))
    print(np.max(np.abs(rec.ints[1] - x_mats[4])))
    print(np.max(np.abs(rec.ints[5] - x_mats[5])))


if __name__ == "__main__":
    main()
