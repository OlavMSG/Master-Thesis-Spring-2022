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


def make_edge(n):
    """
    Make the edge

    Parameters
    ----------
    n : int
        Number of nodes in each spatial direction (_n^2 total nodes).

    Returns
    -------
    edge : np.array
        Index list of all nodal points on the outer edge.

    """
    n2 = n * n
    # Generating nodal points on outer _edge.
    south_edge = np.array([np.arange(1, n), np.arange(2, n + 1)]).T
    east_edge = np.array([np.arange(n, n2 - n + 1, n), np.arange(2 * n, n2 + 1, n)]).T
    north_edge = np.array([np.arange(n2, n2 - n + 1, -1), np.arange(n2 - 1, n2 - n, -1)]).T
    west_edge = np.array([np.arange(n2 - n + 1, n - 1, -n), np.arange(n2 - 2 * n + 1, 0, -n)]).T
    edge = np.vstack((south_edge, east_edge, north_edge, west_edge))

    # Added this to get this script too work.
    edge -= 1
    return edge


def make_p(a, b, n):
    """
    Get the points

    Parameters
    ----------
    n : int
        Number of nodes in each spatial direction (_n^2 total nodes).
    a : float, optional
        Lower limit for x and y. The default is -1.
    b : float, optional
        Upper limit for x and y. The default is 1.

    Returns
    -------
    p : np.array
        Nodal points, (x,y)-coordinates for point i given in row i.
    """
    # Defining auxiliary variables.
    l = np.linspace(a, b, n)
    y, x = np.meshgrid(l, l)

    # Generating nodal points.
    n2 = n * n
    p = np.zeros((n2, 2))
    p[:, 0] = x.T.ravel()
    p[:, 1] = y.T.ravel()
    return p


def getPlateTri(n, a=0, b=1):
    """
    Get the plate (a,b)^2, Triangular elements

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

    n12 = (n - 1) * (n - 1) * 2
    tri = np.zeros((n12, 3), dtype=int)

    def index_map(i, j):
        return i + n * j

    k = 0
    for i in range(n - 1):
        for j in range(n - 1):
            tri[k, 0] = index_map(i, j)
            tri[k, 1] = index_map(i + 1, j)
            tri[k, 2] = index_map(i + 1, j + 1)
            k += 1
            tri[k, 0] = index_map(i, j)
            tri[k, 1] = index_map(i + 1, j + 1)
            tri[k, 2] = index_map(i, j + 1)
            k += 1

    arg = np.argsort(tri[:, 0])
    tri = tri[arg]
    edge = make_edge(n)
    return p, tri, edge


def getPlateRec(n, a=0, b=1):
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
