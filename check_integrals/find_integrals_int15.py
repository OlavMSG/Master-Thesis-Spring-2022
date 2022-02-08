# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from sympy import symbols, simplify, factor, flatten, S, latex
from sympy.matrices import Matrix
from sympy.physics.quantum import TensorProduct

from index_functions import *

x1, x2 = symbols("x1, x2")
x_vec = Matrix([x1, x2])
lx, ly = symbols("L_x, L_y")
u11, u12, u21, u22 = symbols("u11, u12, u21, u22")
nabla_u = Matrix([[u11, u21],
                  [u12, u22]])
v11, v12, v21, v22 = symbols("v11, v12, v21, v22")
nabla_v = Matrix([[v11, v21],
                  [v12, v22]])
a, b = symbols("a, b")
a1, a2, a3, a4 = symbols("a_1, a_2, a_3, a_4")
b1, b2, b3, b4 = symbols("b_1, b_2, b_3, b_4")

mu1, mu2 = symbols("mu1, mu2")

subs_dict = {x1: symbols("\\Tilde{x}_1"),
             x2: symbols("\\Tilde{x}_2"),
             mu1: symbols("\\mu_1"),
             mu2: symbols("\\mu_2"),
             u11: symbols("\\frac{\\partial\\Tilde{u}_1}{\\partial\\Tilde{x}_1}"),
             u12: symbols("\\frac{\\partial\\Tilde{u}_1}{\\partial\\Tilde{x}_2}"),
             u21: symbols("\\frac{\\partial\\Tilde{u}_2}{\\partial\\Tilde{x}_1}"),
             u22: symbols("\\frac{\\partial\\Tilde{u}_2}{\\partial\\Tilde{x}_2}"),
             v11: symbols("\\frac{\\partial\\Tilde{v}_1}{\\partial\\Tilde{x}_1}"),
             v12: symbols("\\frac{\\partial\\Tilde{v}_1}{\\partial\\Tilde{x}_2}"),
             v21: symbols("\\frac{\\partial\\Tilde{v}_2}{\\partial\\Tilde{x}_1}"),
             v22: symbols("\\frac{\\partial\\Tilde{v}_2}{\\partial\\Tilde{x}_2}")}


def print_ints(ints, unique_z, det_j=None, print_latex=False):
    if det_j is None:
        det_j_string = ""
        det_j_latex1 = ""
        det_j_latex2 = ""
        det_j_latex3 = ""
    else:
        det_j_string = " / det(j)"
        det_j_latex1 = "\\frac{"
        det_j_latex2 = "}{|J|}"
        det_j_latex3 = "1"
    for n, intn in enumerate(ints):
        print("int", n + 1)
        print("{", intn, "}" + det_j_string)
        print("=")
        latex_string = ""
        if intn.equals(S.Zero):
            print("+{", S.Zero, "}")
        else:
            for comp in unique_z:
                comp = simplify(comp)
                if comp.is_constant():
                    pass
                else:
                    form = intn.coeff(comp, 1)
                    intn -= comp * form
                    if form.equals(S.Zero):
                        pass
                    else:
                        print("+ {", form, "} * (", comp, ")" + det_j_string)
                        if print_latex:
                            if len(latex_string) == 0:
                                latex_string += f"\\displaystyle \n I_{n+1}="
                            else:
                                latex_string += "\\\\\\\\ \n \\displaystyle \n +"
                            latex_string += "\\int_{\\Tilde{\\Omega}}\n" \
                                            + det_j_latex1 + pplatex(comp) + det_j_latex2 + "\n"  \
                                            + "\\left(" + pplatex(form) + "\\right)" \
                                            + "\n \\,d \\Tilde{\\Omega} \n"
                    if intn.equals(S.Zero):
                        break
            if not intn.equals(S.Zero):
                form = intn.coeff(S.One, 1)
                print("+ 1 * (", form, ")" + det_j_string)
                if print_latex:
                    if len(latex_string) == 0:
                        latex_string += f"\\displaystyle \n I_{n + 1}="
                    else:
                        latex_string += "\\\\\\\\ \n \\displaystyle \n +"
                    latex_string += "\\int_{\\Tilde{\\Omega}}\n" \
                                    + det_j_latex1 + det_j_latex3 + det_j_latex2 \
                                    + "\\left(" + pplatex(form) + "\\right)" \
                                    + "\n \\,d \\Tilde{\\Omega} \n"
            if print_latex:
                print("LATEX:")
                print(latex_string)
            if det_j is not None:
                print("det(j) =", factor(det_j))
                if print_latex:
                    print(pplatex(factor(det_j)))
        print("")


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


def print_sympy_matrix(mat):
    n = mat.shape[0]
    for i in range(n):
        print(simplify(mat[i, :]))


def pplatex(expr):
    return latex(expr.evalf(subs=subs_dict))


def compute_terms(phi_func, scale_det_j=True, print_latex=False):
    phi = phi_func()
    print("phi")
    print_sympy_matrix(phi)
    if print_latex:
        print(pplatex(phi))
    j_mat = phi.jacobian(x_vec)
    print("Jacobian")
    print_sympy_matrix(j_mat)
    if print_latex:
        print(pplatex(j_mat))
    det_j = j_mat.det()
    j_inv = j_mat.inv()
    z_mat = TensorProduct(j_inv, j_inv) * det_j
    ints = get_from_int_1_5_terms(z_mat)
    if scale_det_j:
        print("Jacoian inverse * det(j)")
        print_sympy_matrix(j_inv * det_j)
        if print_latex:
            print(pplatex(simplify(j_inv * det_j)))
        print("Z * det(j)")
        print_sympy_matrix(z_mat * det_j)
        if print_latex:
            print(pplatex(z_mat * det_j))
        unique_z = Matrix(list(set(flatten(z_mat * det_j))))
        print("det(j):")
        print(factor(det_j))
        for i in range(5):
            ints[i] = simplify(ints[i] * det_j)
        print("Unique values in Z * det(j): count:", len(unique_z))
        print(unique_z)
        if print_latex:
            print(pplatex(unique_z))
        print_ints(ints, unique_z, det_j=det_j, print_latex=print_latex)
    else:
        print("Jacoian inverse")
        print_sympy_matrix(j_inv)
        if print_latex:
            print(pplatex(j_inv))
        z_mat = TensorProduct(j_inv, j_inv) * det_j
        print("Z")
        print_sympy_matrix(z_mat)
        if print_latex:
            print(pplatex(z_mat))
        unique_z = Matrix(list(set(flatten(z_mat))))
        print("det(j):")
        print(factor(det_j))
        print("Unique values in Z: count:", len(unique_z))
        print(unique_z)
        if print_latex:
            print(pplatex(unique_z))
        print_ints(ints, unique_z, print_latex=print_latex)
    print("-" * 40)


def scaling_of_rectangle():
    print("-" * 40)
    print("Scaling of Rectangle:")
    print("-" * 40)
    phi = Matrix([
        lx / 2 * x1 + lx / 2,
        ly / 2 * x2 + ly / 2
    ])
    print(phi.evalf(subs={x1: -1, x2: -1}))
    print(phi.evalf(subs={x1: 1, x2: -1}))
    print(phi.evalf(subs={x1: 1, x2: 1}))
    print(phi.evalf(subs={x1: -1, x2: 1}))
    print("-" * 40)
    return phi


def dragging_one_corner_of_rectangle():
    print("-" * 40)
    print("Dragging One Corner of Rectangle:")
    print("-" * 40)
    phi = Matrix([
        x1 + mu1 * x1 * x2,
        x2 + mu2 * x1 * x2
    ])

    print(phi.evalf(subs={x1: 0, x2: 0, mu1: a - 1, mu2: b - 1}))
    print(phi.evalf(subs={x1: 1, x2: 0, mu1: a - 1, mu2: b - 1}))
    print(phi.evalf(subs={x1: 1, x2: 1, mu1: a - 1, mu2: b - 1}))
    print(phi.evalf(subs={x1: 0, x2: 1, mu1: a - 1, mu2: b - 1}))
    print("-" * 40)
    return phi


def dragging_all_corners_of_rectangle():
    print("-" * 40)
    print("Dragging All Corners of Rectangle:")
    print("-" * 40)
    phi = Matrix([
        x1 + a1 * (1. - x1) * (1. - x2) + (a2 - 1.) * x1 * (1. - x2) + (a3 - 1.) * x1 * x2 + a4 * (1. - x1) * x2,
        x2 + b1 * (1. - x1) * (1. - x2) + b2 * x1 * (1. - x2) + (b3 - 1.) * x1 * x2 + (b4 - 1) * (1. - x1) * x2
    ])
    print(phi.evalf(subs={x1: 0., x2: 0.}))
    print(phi.evalf(subs={x1: 1., x2: 0.}))
    print(phi.evalf(subs={x1: 1., x2: 1.}))
    print(phi.evalf(subs={x1: 0., x2: 1.}))
    print("-" * 40)
    return phi


def main():
    choice = -1
    while choice not in (0, 1, 2, 3):
        print("Please choose (0-3)")
        print("0: End script")
        print("1: Scaling of rectangle")
        print("2: Dragging one corner of rectangle")
        print("3: Dragging all corners of rectangle")
        print("4: Scaling of rectangle LATEX")
        print("5: Dragging one corner of rectangle LATEX")
        print("6: Dragging all corners of rectangle LATEX")
        print("Multi run write choice after each other, e.g 1234560 for all options then ending the script")
        choice = input("Run (0-6): ")
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
            elif choice == 4:
                compute_terms(scaling_of_rectangle, scale_det_j=False, print_latex=True)
            elif choice == 5:
                compute_terms(dragging_one_corner_of_rectangle, print_latex=True)
            elif choice == 6:
                compute_terms(dragging_all_corners_of_rectangle, print_latex=True)
            else:
                print("Choice is not valid please try again")

            choice = -1


if __name__ == '__main__':
    main()
