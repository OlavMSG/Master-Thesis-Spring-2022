# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from sympy import symbols, expand, S
from sympy.matrices import Matrix, zeros
from sympy.physics.quantum import TensorProduct
from old_no_longer_in_use.index_functions import *

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

cx1, cy1 = symbols("cx1, cy1")
cx2, cy2 = symbols("cx2, cy2")
cx3, cy3 = symbols("cx3, cy3")

subs_mat = Matrix([[cx1, cy1],
                   [cx2, cy2],
                   [cx3, cy3]])


def inv_indexmap(k):
    return k // 2, k % 2


def print_uv(ddot_uv, nabla_u=nabla_u, nabla_v=nabla_v):
    print("nabla u : nabla v")
    text = ""
    for i in range(2):
        ui = nabla_u[i, :]
        for j in range(2):
            vj = nabla_v[j, :]
            for k in range(2):
                if ui[k].equals(vj[k]):
                    form = expand(ui[k] ** 2 * ddot_uv.coeff(ui[k], 2))
                else:
                    form = expand(ui[k] * ddot_uv.coeff(ui[k], 1).coeff(vj[k], 1) * vj[k])
                print("+{", form, "}")
                if k == 1:
                    print("\n")


def print_uvt(ddot_uvt, nabla_u=nabla_u, nabla_v=nabla_v):
    print("nabla u : nabla v.T")
    for i in range(2):
        ui = nabla_u[i, :]
        for j in range(2):
            vj = nabla_v[j, :]
            for k in range(2):
                for n in range(2):
                    if ui[k].equals(vj[n]):
                        form = expand(ui[k] ** 2 * ddot_uvt.coeff(ui[k], 2))
                    else:
                        form = expand(ui[k] * ddot_uvt.coeff(ui[k], 1).coeff(vj[n], 1) * vj[n])
                    print("+{", form, "}")
                    if n == 1 and k == 1:
                        print("\n")


def print_div_uv(div_uv, nabla_u=nabla_u, nabla_v=nabla_v):
    print("div u : div v")
    for i in range(2):
        ui = nabla_u[i, :]
        for j in range(2):
            vj = nabla_v[j, :]
            for k in range(2):
                for n in range(2):
                    if ui[k].equals(vj[n]):
                        form = expand(ui[k] ** 2 * div_uv.coeff(ui[k], 2))
                    else:
                        form = expand(ui[k] * div_uv.coeff(ui[k], 1).coeff(vj[n], 1) * vj[n])
                    print("+{", form, "}")
                    if n == 1 and k == 1:
                        print("\n")


def get_from_int_1_5_terms(z_mat):
    int1 = S.Zero
    int2 = S.Zero
    int3 = S.Zero
    int4 = S.Zero
    int5 = S.Zero
    for i in range(2):
        for j in range(2):
            int3 += nabla_u[j, i] * z_mat[k_hat(j), n_hat(i)] * nabla_v[j, l_hat(i)]
            int4 += nabla_u[j, i] * z_mat[m_hat(i), n_hat(j)] * nabla_v[l_hat(j), l_hat(i)]
            int5 += nabla_u[j, i] * z_mat[m_hat(i), m_hat(j)] * nabla_v[l_hat(j), l_hat(i)]
            for k in range(2):
                int1 += nabla_u[j, i] * z_mat[i_hat(j, k), k_hat(i)] * nabla_v[k, i]
                int2 += nabla_u[j, i] * z_mat[i_hat(j, k), h_hat(i)] * nabla_v[k, i]

    ints = [int1, int2, int3, int4, int5]
    return ints


def not_d(d):
    if d == 0:
        return 1
    else:
        return 0


def print_sym_matrix(mat):
    n = mat.shape[0]
    for i in range(n):
        print(mat[i, :])


def check_sym_ints(ints, m):
    a_list = []
    for n, intn in enumerate(ints):
        print("\nint", n + 1)
        a = zeros(m, m)
        for ki in range(m):
            i, di = inv_indexmap(ki)
            not_di = not_d(di)
            for kj in range(m):
                j, dj = inv_indexmap(kj)
                not_dj = not_d(dj)
                uv_list = [nabla_u[0, di], nabla_u[1, di], nabla_v[0, dj], nabla_v[1, dj],
                           nabla_u[0, not_di], nabla_u[1, not_di], nabla_v[0, not_dj], nabla_v[1, not_dj]]
                c_list = [subs_mat[i, 0], subs_mat[i, 1], subs_mat[j, 0], subs_mat[j, 1],
                          0., 0., 0., 0.]
                subs_dict = dict(zip(uv_list, c_list))
                a[ki, kj] = intn.evalf(subs=subs_dict)
                # print("(i, di)", (i, di), "(j, dj)", (j, dj), "(ki, kj)", (ki, kj))
        print_sym_matrix(a)
        print("Is symmetric:", a.equals(a.T))
        a_list.append(a)


def main():
    m = 4
    ints = get_from_int_1_5_terms(z_mat)
    check_sym_ints(ints, m)


if __name__ == '__main__':
    main()
