# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from sympy import symbols, simplify, factor, S
from sympy.matrices import Matrix
from sympy.physics.quantum import TensorProduct

from find_integrals_int15 import scaling_of_rectangle, dragging_one_corner_of_rectangle, \
    dragging_all_corners_of_rectangle
from old_no_longer_in_use.index_functions import *

x1, x2 = symbols("x1, x2")
x_vec = Matrix([x1, x2])
lx, ly = symbols("Lx, Ly")
u11, u12, u21, u22 = symbols("u11, u12, u21, u22")
nabla_u = Matrix([[u11, u21],
                  [u12, u22]])
v11, v12, v21, v22 = symbols("v11, v12, v21, v22")
nabla_v = Matrix([[v11, v21],
                  [v12, v22]])
a, b = symbols("a, b")
a1, a2, a3, a4 = symbols("a1, a2, a3, a4")
b1, b2, b3, b4 = symbols("b1, b2, b3, b4")


def print_uv(ddot_uv):
    # ddot_uv = simplify(cancel(ddot_uv))
    print("nabla u : nabla v")
    for i in range(2):
        ui = nabla_u[i, :]
        for j in range(2):
            vj = nabla_v[j, :]
            for k in range(2):
                form = ui[k] * ddot_uv.coeff(ui[k], 1).coeff(vj[k], 1) * vj[k]
                print("+{", form, "}")
                if k == 1:
                    print("\n")


def print_uvt(ddot_uvt):
    # ddot_uvt = simplify(cancel(ddot_uvt))
    print("nabla u : nabla v.T")
    for i in range(2):
        ui = nabla_u[i, :]
        for j in range(2):
            vj = nabla_v[j, :]
            for k in range(2):
                for n in range(2):
                    form = ui[k] * ddot_uvt.coeff(ui[k], 1).coeff(vj[n], 1) * vj[n]
                    print("+{", form, "}")
                    if k == 1 and n == 1:
                        print("\n")


def print_div_uv(div_uv):
    # div_uv = simplify(cancel(div_uv))
    print("div u : div v")
    for i in range(2):
        ui = nabla_u[i, :]
        for j in range(2):
            vj = nabla_v[j, :]
            for k in range(2):
                for n in range(2):
                    form = ui[k] * div_uv.coeff(ui[k], 1).coeff(vj[n], 1) * vj[n]
                    print("+{", form, "}")
                    if k == 1 and n == 1:
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


def compute_from_ints(int1, int2, int3, int4, int5, scale=None):
    if scale is None:
        ddot_uv = int1 + int2
        ddot_uvt = int1 + int3 + int4
        div_uv = int1 + int3 + int5
    else:
        ddot_uv = simplify((int1 + int2) * scale)
        ddot_uvt = simplify((int1 + int3 + int4) * scale)
        div_uv = simplify((int1 + int3 + int5) * scale)
    return ddot_uv, ddot_uvt, div_uv


def print_sympy_matrix(mat):
    n = mat.shape[0]
    for i in range(n):
        print(simplify(mat[i, :]))


def compute_terms(phi_func, scale_det_j=True):
    phi = phi_func()
    j_mat = phi.jacobian(x_vec)
    print("Jacobian")
    print_sympy_matrix(j_mat)
    det_j = j_mat.det()
    j_inv = j_mat.inv()
    z_mat = TensorProduct(j_inv, j_inv) * det_j
    ints = get_from_int_1_5_terms(z_mat)
    if scale_det_j:
        print("Jacoian inverse * det(j)")
        print_sympy_matrix(j_inv * det_j)
        print("Z * det(j)")
        print_sympy_matrix(z_mat * det_j)
        print("det(j):")
        print(factor(det_j))
        for i in range(5):
            print("int", i + 1, " * det(j)\n", simplify(ints[i] * det_j))
        ddot_uv, ddot_uvt, div_uv = compute_from_ints(*ints, scale=det_j)
        print_uv(ddot_uv)
        print("Divide by det(j):\n", det_j)
        print_uvt(ddot_uvt)
        print("Divide by det(j):\n", det_j)
        print_div_uv(div_uv)
        print("Divide by det(j):\n", det_j)
    else:
        print("Jacoian inverse")
        print_sympy_matrix(j_inv)
        z_mat = TensorProduct(j_inv, j_inv) * det_j
        print("Z")
        print_sympy_matrix(z_mat)
        print("det(j):")
        print(factor(det_j))
        ddot_uv, ddot_uvt, div_uv = compute_from_ints(*ints)
        for i in range(5):
            print("int", i + 1, "\n", ints[i])
        print_uv(ddot_uv)
        print_uvt(ddot_uvt)
        print_div_uv(div_uv)
    print("-" * 40)


def main():
    choice = -1
    while choice not in (0, 1, 2, 3):
        print("Please choose (0-3)")
        print("0: End script")
        print("1: Scaling of rectangle")
        print("2: Dragging one corner of rectangle")
        print("3: Dragging all corners of rectangle")
        print("Multi run write choice after each other, e.g 1230 for all options then ending the script")
        choice = input("Run (0-3): ")
        for choice in list(choice):
            try:
                choice = int(choice)
            except ValueError:
                choice = -1

            if choice == 0:
                break
            elif choice == 1:
                compute_terms(scaling_of_rectangle, scale_det_j=False)
            elif choice == 2:
                compute_terms(dragging_one_corner_of_rectangle)
            elif choice == 3:
                compute_terms(dragging_all_corners_of_rectangle)
            else:
                print("Choice is not valid please try again")

            choice = -1


if __name__ == '__main__':
    main()
