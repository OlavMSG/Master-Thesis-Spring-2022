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

    # a1_full, a2_full = rec.compute_a1_and_a2_from_ints()
    # a_full = compute_a(e_young, nu_poisson, a1_full, a2_full)
    print("-" * 50)
    s = perf_counter()
    x_mats = matrix_least_squares(m, rec, mls_mode="sparse")
    print("time matrix least squares:", perf_counter() - s)
    print("checking np.max(np.abs(...))")

    print("mu part")
    print(np.max(np.abs(rec.ints[0] + 0.5 * rec.ints[2] - x_mats[0])))
    print(np.max(np.abs(rec.ints[1] + 0.5 * rec.ints[3] - x_mats[1])))
    print(np.max(np.abs(0.5 * rec.ints[4] - x_mats[2])))
    print("lambda part")
    print(np.max(np.abs(rec.ints[0] - x_mats[3])))
    print(np.max(np.abs(rec.ints[1] - x_mats[4])))
    print(np.max(np.abs(rec.ints[5] - x_mats[5])))

    print("-" * 50)
    s = perf_counter()
    x_mats = matrix_least_squares(m, rec, mls_mode="array", compute_a_str="a1")
    print("time matrix least squares:", perf_counter() - s)
    print("checking np.max(np.abs(...))")

    print("do only mu part")
    print(np.max(np.abs(rec.ints[0] + 0.5 * rec.ints[2] - x_mats[0])))
    print(np.max(np.abs(rec.ints[1] + 0.5 * rec.ints[3] - x_mats[1])))
    print(np.max(np.abs(0.5 * rec.ints[4] - x_mats[2])))

    print("-"*50)
    s = perf_counter()
    x_mats = matrix_least_squares(m, rec, mls_mode="array", compute_a_str="a2")
    print("time matrix least squares:", perf_counter() - s)
    print("checking np.max(np.abs(...))")

    print("do only lambda part")
    print(np.max(np.abs(rec.ints[0] - x_mats[0])))
    print(np.max(np.abs(rec.ints[1] - x_mats[1])))
    print(np.max(np.abs(rec.ints[5] - x_mats[2])))



if __name__ == "__main__":
    main()