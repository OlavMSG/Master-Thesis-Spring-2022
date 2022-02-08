# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np
from time import perf_counter


def lagrange_1D_const(x1, x2):
    c = 1 / (x2 - x1)
    lc = np.array([-c, x2 * c])
    return lc


def lagrange_2D_mat(x_vec, y_vec):
    pc_mat = np.array([[1, -1, 1, -1],
                       [-y_vec[2], y_vec[3], -y_vec[0], y_vec[1]],
                       [-x_vec[2], x_vec[3], -x_vec[0], x_vec[1]],
                       [x_vec[2] * y_vec[2], -x_vec[3] * y_vec[3], x_vec[0] * y_vec[0], -x_vec[1] * y_vec[1]]]) / (
                         (x_vec[1] - x_vec[0]) * (y_vec[3] - y_vec[0]))

    """lxc1 = lagrange_1D_const(x_vec[0], x_vec[1])
    lxc2 = lagrange_1D_const(x_vec[1], x_vec[0])
    lyc1 = lagrange_1D_const(y_vec[0], y_vec[3])
    lyc2 = lagrange_1D_const(y_vec[3], y_vec[0])
    pc_mat = np.zeros((4, 4))

    pc_mat[:, 0] = np.kron(lxc1, lyc1)
    pc_mat[:, 1] = np.kron(lxc2, lyc1)
    pc_mat[:, 2] = np.kron(lxc2, lyc2)
    pc_mat[:, 3] = np.kron(lxc1, lyc2)"""

    return pc_mat


def lagrange_1D_func(lc):
    def l_func(x):
        return lc[0] * x + lc[1]

    return l_func


def lagrange_2D_func(pc):
    def p_func(x, y):
        return pc[0] * x * y + pc[1] * x + pc[2] * y + pc[3]

    return p_func


def check_1d():
    x1, x2 = 0, 1
    l1c = lagrange_1D_const(x1, x2)
    l2c = lagrange_1D_const(x2, x1)
    l1 = lagrange_1D_func(l1c)
    l2 = lagrange_1D_func(l2c)
    print(l1(x1), l1(x2))
    print(l1c)
    print(l2(x1), l2(x2))
    print(l2c)


def check1_2d():
    l = np.linspace(0, 1, 2)
    y, x = np.meshgrid(l, l)
    x_vec, y_vec = x.T.ravel(), y.T.ravel()
    arg = [0, 1, 3, 2]
    x_vec = x_vec[arg]
    y_vec = y_vec[arg]
    print(x_vec)
    print(y_vec)
    pc_mat = lagrange_2D_mat(x_vec, y_vec)
    print(pc_mat)
    for i in range(4):
        print(i)
        pc = pc_mat[: i]
        p_func = lagrange_2D_func(pc)
        for j in range(4):
            print(x_vec[j], y_vec[j])
            print(p_func(x_vec[j], y_vec[j]))


def check2_2d():
    l = np.linspace(4, 6, 2)
    l2 = np.linspace(2, 4, 2)
    y, x = np.meshgrid(l, l2)
    x_vec, y_vec = x.T.ravel(), y.T.ravel()
    arg = [0, 1, 3, 2]
    x_vec = x_vec[arg]
    y_vec = y_vec[arg]
    print(x_vec)
    print(y_vec)
    s = perf_counter()
    pc_mat = lagrange_2D_mat(x_vec, y_vec)
    print("time.", perf_counter() - s)
    pk = np.linalg.inv(pc_mat)
    print("pc_mat")
    print(pc_mat)
    print("pk")
    print(pk)
    s = perf_counter()
    mk = np.column_stack((x_vec * y_vec, x_vec, y_vec, np.ones(4)))
    ck = np.linalg.inv(mk)
    print("time.", perf_counter() - s)
    print("ck")
    print(ck)
    print("mk")
    print(mk)


def main():
    check2_2d()


if __name__ == '__main__':
    main()
