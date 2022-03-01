# -*- coding: utf-8 -*-

"""
@author: Olav M.S. Gran
"""
from sympy import symbols, expand, S
from sympy.matrices import Matrix
from sympy.physics.quantum import TensorProduct

from find_integrals_uv import print_div_uv, print_uvt, print_uv, get_from_int_1_5_terms, compute_from_ints
from index_functions import *
import numpy as np

x1, x2 = symbols("x1, x2")
x_vec = Matrix([x1, x2])
u11, u12, u21, u22 = symbols("u11, u12, u21, u22")
nabla_u = Matrix([[u11, u21],
                  [u12, u22]])
v11, v12, v21, v22 = symbols("v11, v12, v21, v22")
nabla_v = Matrix([[v11, v21],
                  [v12, v22]])

j11, j12, j21, j22 = symbols("j11, j12, j21, j22")
j_inv = Matrix([[j11, j12],
                [j21, j22]])

z_mat = TensorProduct(j_inv, j_inv)

nabla_u_real = j_inv.T * nabla_u
nabla_v_real = j_inv.T * nabla_v


def el_mult_2d(u, v):
    return Matrix([[u[0, 0] * v[0, 0], u[0, 1] * v[0, 1]],
                   [u[1, 0] * v[1, 0], u[1, 1] * v[1, 1]]])


def double_dot_2d(u, v):
    return sum(el_mult_2d(u, v))


def check_from_real():
    print("-" * 30)
    print("Check integrals terms form real")
    ddot_uv = expand(double_dot_2d(nabla_u_real, nabla_v_real))
    print_uv(ddot_uv)
    ddot_uvt = expand(double_dot_2d(nabla_u_real, nabla_v_real.T))
    print_uvt(ddot_uvt)
    div_uv = expand(nabla_u_real.trace() * nabla_v_real.trace())
    print_div_uv(div_uv)
    print("Unique terms:")
    terms = str(ddot_uv).split("+") + str(ddot_uvt).split("+") + str(div_uv).split("+")
    u_terms = np.unique(terms)
    print(len(u_terms), "of", len(terms))
    print("-" * 30)


def get_ddot_uv():
    ddot_uv = S.Zero
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for n in range(2):
                    ddot_uv += nabla_u[j, i] * z_mat[i_hat(j, k), j_hat(n, n)] * nabla_v[k, i]
    return ddot_uv


def get_ddot_uvt():
    ddot_uvt = S.Zero
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for n in range(2):
                    ddot_uvt += nabla_u[k, i] * z_mat[i_hat(k, n), j_hat(i, j)] * nabla_v[n, j]
    return ddot_uvt


def get_div_uv():
    div_uv = S.Zero
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for n in range(2):
                    div_uv += nabla_u[k, i] * z_mat[i_hat(k, n), i_hat(i, j)] * nabla_v[n, j]
    return div_uv


def check_from_z():
    print("-" * 30)
    print("Check integrals terms form z_mat_funcs")
    ddot_uv_true = expand(double_dot_2d(nabla_u_real, nabla_v_real))
    ddot_uv = get_ddot_uv()
    eq1 = ddot_uv.equals(ddot_uv_true)
    print("Is equal: ", eq1)
    print("from real:")
    print_uv(ddot_uv_true)
    print("from Z:")
    print_uv(ddot_uv)
    ddot_uvt_true = expand(double_dot_2d(nabla_u_real, nabla_v_real.T))
    ddot_uvt = get_ddot_uvt()
    eq2 = ddot_uvt.equals(ddot_uvt_true)
    print("Is equal: ", eq2)
    print("from real:")
    print_uvt(ddot_uvt_true)
    print("from Z:")
    print_uvt(ddot_uvt)
    div_uv_true = expand(nabla_u_real.trace() * nabla_v_real.trace())
    div_uv = get_div_uv()
    eq3 = div_uv.equals(div_uv_true)
    print("Is equal: ", eq3)
    print("from real:")
    print_div_uv(div_uv_true)
    print("from Z:")
    print_div_uv(div_uv)
    print("-" * 30)


def check_from_ints():
    print("-" * 30)
    print("Check integrals terms form ints 1-5")
    ints = get_from_int_1_5_terms(z_mat)
    ddot_uv, ddot_uvt, div_uv = compute_from_ints(*ints)
    ddot_uv_true = expand(double_dot_2d(nabla_u_real, nabla_v_real))
    eq1 = ddot_uv.equals(ddot_uv_true)
    print("Is equal: ", eq1)
    print("from real:")
    print_uv(ddot_uv_true)
    print("from ints:")
    print_uv(ddot_uv)
    ddot_uvt_true = expand(double_dot_2d(nabla_u_real, nabla_v_real.T))
    eq2 = ddot_uvt.equals(ddot_uvt_true)
    print("Is equal: ", eq2)
    print("from real:")
    print_uvt(ddot_uvt_true)
    print("from ints:")
    print_uvt(ddot_uvt)
    div_uv_true = expand(nabla_u_real.trace() * nabla_v_real.trace())
    eq3 = div_uv.equals(div_uv_true)
    print("Is equal: ", eq3)
    print("from real:")
    print_div_uv(div_uv_true)
    print("from ints:")
    print_div_uv(div_uv)
    print("-" * 30)


def main():
    check_from_real()
    check_from_z()
    check_from_ints()


if __name__ == '__main__':
    main()