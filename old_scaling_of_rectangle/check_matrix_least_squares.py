# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from time import perf_counter

from scipy.sparse.linalg import norm

from old_scaling_of_rectangle.scalable_rectangle_class import ScalableRectangle
from old_scaling_of_rectangle.matrix_least_squares import matrix_least_squares


def main():
    n = 2
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
    x_mats = matrix_least_squares(m, rec, mls_mode="sp")
    print("time matrix least squares:", perf_counter() - s)
    print("checking np.linalg.norm(...)")

    print("mu part")
    print(norm(rec.ints[0] + 0.5 * rec.ints[2] - x_mats[0]))
    print(norm(rec.ints[1] + 0.5 * rec.ints[3] - x_mats[1]))
    print(norm(0.5 * rec.ints[4] - x_mats[2]))
    print("lambda part")
    print(norm(rec.ints[0] - x_mats[3]))
    print(norm(rec.ints[1] - x_mats[4]))
    print(norm(rec.ints[5] - x_mats[5]))
    print("Extra part 1, 2*mu*lambda, lx*ly")
    print(norm(x_mats[6]))
    print(norm(x_mats[7]))
    print(norm(x_mats[8]))


    print("-" * 50)
    m = 5
    s = perf_counter()
    x_mats = matrix_least_squares(m, rec, mls_mode="sp", compute_a_str="a1")
    print("time matrix least squares:", perf_counter() - s)
    print("checking np.linalg.norm(...)")

    print("do only mu part")
    print(norm(rec.ints[0] + 0.5 * rec.ints[2] - x_mats[0]))
    print(norm(rec.ints[1] + 0.5 * rec.ints[3] - x_mats[1]))
    print(norm(0.5 * rec.ints[4] - x_mats[2]))

    print("-"*50)
    s = perf_counter()
    x_mats = matrix_least_squares(m, rec, mls_mode="sp", compute_a_str="a2")
    print("time matrix least squares:", perf_counter() - s)
    print("checking norm(...)")

    print("do only lambda part")
    print(norm(rec.ints[0] - x_mats[0]))
    print(norm(rec.ints[1] - x_mats[1]))
    print(norm(rec.ints[5] - x_mats[2]))



if __name__ == "__main__":
    main()
