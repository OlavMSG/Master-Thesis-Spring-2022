# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
based on Specialization-Project-fall-2021
"""

import numpy as np

from gauss_quadrature import line_integral_with_basis
from helpers import expand_index, VectorizedFunction2D


def assemble_f_neumann(n, p, neumann_edge, neumann_bc_func):
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
        f_load_neumann[expanded_ek] += line_integral_with_basis(*p[ek, :], 4, neumann_bc_func)
    return f_load_neumann
