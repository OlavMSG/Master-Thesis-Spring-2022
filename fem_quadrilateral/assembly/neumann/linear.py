# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
based on Specialization-Project-fall-2021
"""

import numpy as np

from fem_quadrilateral.assembly.neumann.gauss_quadrature import line_integral_with_linear_basis
from fem_quadrilateral.helpers import expand_index


def assemble_f_neumann(n, p, neumann_edge, neumann_bc_func, nq=4):
    """
    Assemble the neumann load vector

    Parameters
    ----------
    n : int
        number of node along one axis.
    p : np.array
        list of points.
    neumann_edge : np.array
        array of the edges of the triangulation.
    neumann_bc_func : function, VectorizedFunction2D
        the neumann boundary condition function.
    nq : int, optional
        quadrature scheme order. The default is 4.

    Returns
    -------
    f_load_neumann : np.array
        load vector for neumann.

    """
    n2d = n * n * 2
    # load vector
    f_load_neumann = np.zeros(n2d, dtype=float)
    for ek in neumann_edge:
        # p1 = p[ek[0], :]
        # p2 = p[ek[1], :]
        # expand the index
        expanded_ek = expand_index(ek)
        # add local contribution
        f_load_neumann[expanded_ek] += line_integral_with_linear_basis(*p[ek, :], neumann_bc_func, nq)
    return f_load_neumann
