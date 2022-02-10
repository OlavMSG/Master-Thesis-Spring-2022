# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve

from get_plate import getPlateRec
from assembly import assemble_a1_a2_f
from helpers import VectorizedFunction2D, compute_a, expand_index, get_u_exact
from scaling_of_rectangle.scalable_rectangle_class import ScalableRectangle


def plot_mesh(n, p, tri):
    """
    Plot the mesh

    Parameters
    ----------
    n : int
        number of nodes along the axes.
    p : np.array
        Nodal points, (x,y)-coordinates for point i given in row i.
    tri : np.array
        Elements. Index to the three corners of element i given in row i.

    Returns
    -------
    None.

    """
    def get_line(p1, p2):
        return np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]])

    plt.figure("Mesh plot", figsize=(7, 7))
    plt.title(f"Mesh {n - 1}x{n - 1}")
    for nk in tri:
        p1 = p[nk[0], :]
        p2 = p[nk[1], :]
        p3 = p[nk[2], :]
        p4 = p[nk[3], :]

        plt.plot(*get_line(p1, p2), color="C0")
        plt.plot(*get_line(p2, p3), color="C0")
        plt.plot(*get_line(p3, p4), color="C0")
        plt.plot(*get_line(p4, p1), color="C0")

    plt.grid()


def compute_a1_and_a2_from_ints(z, int11, int12, int21, int22, int4, int5):
    # ints = (int11, int12, int21, int22, int4, int5)
    # int 3 = 0

    int1 = z[0, 0] * int11 + z[3, 3] * int12
    int2 = z[0, 0] * int21 + z[3, 3] * int22

    a1 = int1 + 0.5 * (int2 + int4)
    a2 = int1 + int5

    return a1, a2


def f_func(x, y):
    return 0, 0


def u_exact_func(x, y):
    return x, 0.


def main():
    n = 2
    lx, ly = 3, 1
    rec = ScalableRectangle(lx, ly)
    e_young, nu_poisson = 2.1e5, 0.3
    tol = 1e-14

    n += 1
    p, tri, edge = getPlateRec(n, -1, 1)
    plot_mesh(n, p, tri)
    plt.show()

    def f_func_comp_phi(x, y):
        return f_func(*rec.phi(x, y))

    f_func1 = VectorizedFunction2D(f_func_comp_phi)
    u_exact = get_u_exact(p, u_exact_func)

    s = perf_counter()
    ints, f_load_lv_full = assemble_a1_a2_f(n, p, tri, f_func1, True)
    print("time:", perf_counter() - s)
    a1_full, a2_full = compute_a1_and_a2_from_ints(rec.z, *ints)
    a_full = compute_a(e_young, nu_poisson, a1_full, a2_full)

    dirichlet_edge_index = np.unique(edge)
    # free index is unique index minus dirichlet edge index
    free_index = np.setdiff1d(tri, dirichlet_edge_index)
    expanded_free_index = expand_index(free_index)
    expanded_dirichlet_edge_index = expand_index(dirichlet_edge_index)

    dirichlet_xy_index = np.ix_(expanded_free_index, expanded_dirichlet_edge_index)
    a_dirichlet = a_full[dirichlet_xy_index]

    free_xy_index = np.ix_(expanded_free_index, expanded_free_index)
    a_free = a_full[free_xy_index].tocsr()
    f_load_lv_free = f_load_lv_full[expanded_free_index]

    f_load = f_load_lv_free - a_dirichlet @ u_exact.flatt_values[expanded_dirichlet_edge_index]

    uh_full = np.zeros_like(u_exact.flatt_values)
    uh_full[expanded_free_index] = spsolve(a_free, f_load)
    uh_full[expanded_dirichlet_edge_index] = u_exact.flatt_values[expanded_dirichlet_edge_index]

    # discrete max norm, holds if u_exact is linear (Terms 1, x, y)
    test_res = np.all(np.abs(uh_full - u_exact.flatt_values) < tol)

    print(test_res)


if __name__ == '__main__':
    main()
