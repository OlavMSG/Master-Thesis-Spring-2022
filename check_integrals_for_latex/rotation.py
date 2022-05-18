# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from find_integrals_int15 import scaling_of_rectangle, dragging_one_corner_of_rectangle, \
    dragging_all_corners_of_rectangle, print_sympy_matrix
from sympy import symbols, sin, cos
from sympy.matrices import Matrix
import numpy as np

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

theta = symbols("theta")

rot_mat = Matrix([[cos(theta), - sin(theta)],
                  [sin(theta), cos(theta)]])


def rotate(rec, alpha=None, mode=None):
    phi = rot_mat * rec
    if alpha is not None:
        phi = phi.evalf(subs={theta: alpha})
    if mode == "-1,1":
        print_sympy_matrix(phi.evalf(subs={x1: -1., x2: -1.}))
        print("")
        print_sympy_matrix(phi.evalf(subs={x1: 1., x2: -1.}))
        print("")
        print_sympy_matrix(phi.evalf(subs={x1: 1., x2: 1.}))
        print("")
        print_sympy_matrix(phi.evalf(subs={x1: -1., x2: 1.}))
        print("")
    else:
        print(phi.evalf(subs={x1: 0., x2: 0.}))
        print("")
        print(phi.evalf(subs={x1: 1., x2: 0.}))
        print("")
        print(phi.evalf(subs={x1: 1., x2: 1.}))
        print("")
        print(phi.evalf(subs={x1: 0., x2: 1.}))


def main():
    alpha = np.pi / 2
    rec1 = scaling_of_rectangle()
    rotate(rec1, alpha=alpha, mode="-1,1")
    rec2 = dragging_one_corner_of_rectangle()
    rotate(rec2, alpha=alpha)
    rec3 = dragging_all_corners_of_rectangle()
    rotate(rec3, alpha=alpha)


if __name__ == '__main__':
    main()
