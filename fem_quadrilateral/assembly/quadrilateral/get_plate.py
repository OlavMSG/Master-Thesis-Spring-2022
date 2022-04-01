# -*- coding: utf-8 -*-
# Description:
#   Generate a mesh triangulation of the reference square (a,b)^2.
#
# Arguments:
#    _n      Number of nodes in each spatial direction (_n^2 total nodes).
#    a      Lower limit for x and y
#    b      Upper limit for x and y
#
# Returns:
#   p		Nodal points, (x,y)-coordinates for point i given in row i.
#   tri   	Elements. Index to the three corners of element i given in row i.
#   edge  	Index list of all nodal points on the outer edge.
#
#   Written by Olav M. S. Gran using the
#   Boiler code by: Kjetil a. Johannessen, Abdullah Abdulhaque (October 2019)
#   from https://wiki.math.ntnu.no/tma4220/2020h/project : getplate.py
#
# code taken from Specialization-Project-fall-2021

import numpy as np
from fem_quadrilateral.assembly.get_plate_base import make_p, make_edge


def getPlate(n, a=0, b=1):
    """
    Get the plate (a,b)^2, Rectangle Elements

    Parameters
    ----------
    n : int
        Number of nodes in each spatial direction (_n^2 total nodes).
    a : float, optional
        Lower limit for x and y. The default is 0.
    b : float, optional
        Upper limit for x and y. The default is 1.

    Returns
    -------
    p : np.array
        Nodal points, (x,y)-coordinates for point i given in row i.
    tri : np.array
        Elements. Index to the three corners of element i given in row i.
    edge : np.array
        Index list of all nodal points on the outer edge.

    """
    # Generating nodal points.
    p = make_p(a, b, n)

    # Generating elements.

    n12 = (n - 1) * (n - 1)
    tri = np.zeros((n12, 4), dtype=int)

    def index_map(i, j):
        return i + n * j

    k = 0
    for i in range(n - 1):
        for j in range(n - 1):
            tri[k, 0] = index_map(i, j)
            tri[k, 1] = index_map(i + 1, j)
            tri[k, 2] = index_map(i + 1, j + 1)
            tri[k, 3] = index_map(i, j + 1)
            k += 1

    arg = np.argsort(tri[:, 0])
    tri = tri[arg]
    edge = make_edge(n)
    return p, tri, edge

