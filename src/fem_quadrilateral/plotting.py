# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from .base_solver import BaseSolver


def plot_mesh(solver: BaseSolver, *geo_params: float):
    # set nice plotting
    fontsize = 20
    new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (7, 7),
                  'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
                  'figure.titlesize': fontsize,
                  'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
    plt.rcParams.update(new_params)
    plt.figure("Mesh plot")
    p_non_ref = solver.vectorized_phi(solver.p[:, 0], solver.p[:, 1], *geo_params)
    verts = p_non_ref[solver.tri]
    pc = PolyCollection(verts, color="blue", linewidths=2, facecolor="None")
    ax = plt.gca()
    ax.add_collection(pc)
    ax.autoscale()
    plt.axis("off")
    xb, xt = plt.xlim()
    yb, yt = plt.ylim()
    xy_min = min(xb, yb)
    xy_max = max(xt, yt)
    plt.xlim(xy_min, xy_max)
    plt.ylim(xy_min, xy_max)
    print("Please call plt.show() to show the plot.")


def plot_pod_mode(solver: BaseSolver, i: int):
    # set nice plotting
    fontsize = 20
    new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
                  'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
                  'figure.titlesize': fontsize,
                  'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
    plt.rcParams.update(new_params)
    plt.figure("Mesh plot", figsize=(7, 7))

    pod_mode = solver.rb_pod_mode(i)
    p_non_ref = solver.vectorized_phi(solver.p[:, 0], solver.p[:, 1], *pod_mode.geo_params)
    verts = (p_non_ref + pod_mode.values)[solver.tri]
    pc = PolyCollection(verts, color="blue", linewidths=2, facecolor="None")
    ax = plt.gca()
    ax.add_collection(pc)
    ax.autoscale()
    print("Please call plt.show() to show the plot.")


def plot_hf_displacment(solver: BaseSolver):
    # set nice plotting
    fontsize = 20
    new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
                  'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
                  'figure.titlesize': fontsize,
                  'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
    plt.rcParams.update(new_params)
    plt.figure("HF displacement")

    p_non_ref = solver.vectorized_phi(solver.p[:, 0], solver.p[:, 1], *solver.uh.geo_params)
    ax = plt.gca()
    verts1 = (p_non_ref + solver.uh.values)[solver.tri]
    pc1 = PolyCollection(verts1, linewidths=2, color="lightgray", edgecolor="gray", alpha=0.8)
    ax.add_collection(pc1)
    verts2 = p_non_ref[solver.tri]
    pc2 = PolyCollection(verts2, linewidths=2, color="none", edgecolor="black", alpha=0.23)
    ax.add_collection(pc2)
    ax.autoscale()
    print("Please call plt.show() to show the plot.")


def plot_rb_displacment(solver: BaseSolver):
    # set nice plotting
    fontsize = 20
    new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
                  'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
                  'figure.titlesize': fontsize,
                  'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
    plt.rcParams.update(new_params)
    plt.figure("RB Displacement")

    p_non_ref = solver.vectorized_phi(solver.p[:, 0], solver.p[:, 1], *solver.uh_rom.geo_params)
    ax = plt.gca()
    verts1 = (p_non_ref + solver.uh_rom.values)[solver.tri]
    pc1 = PolyCollection(verts1, linewidths=2, color="lightgray", edgecolor="gray", alpha=0.8)
    ax.add_collection(pc1)
    verts2 = p_non_ref[solver.tri]
    pc2 = PolyCollection(verts2, linewidths=2, color="none", edgecolor="black", alpha=0.23)
    ax.add_collection(pc2)
    ax.autoscale()
    print("Please call plt.show() to show the plot.")


def _quadrilaterals_to_triangles(quads: np.ndarray) -> np.ndarray:
    tris = np.zeros((2 * len(quads), 3))
    for i in range(len(quads)):
        j = 2 * i
        tris[j, :] = quads[i][[0, 1, 2]]
        tris[j + 1, :] = quads[i][[0, 3, 2]]
    return tris


def plot_hf_von_mises(solver: BaseSolver, levels: Optional[np.ndarray] = None):
    fontsize = 20
    new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
                  'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
                  'figure.titlesize': fontsize,
                  'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
    plt.rcParams.update(new_params)
    if levels is None:
        levels = np.linspace(0, np.max(solver.uh.von_mises), 25)
    plt.figure("Hf von mises")
    p_non_ref = solver.vectorized_phi(solver.p[:, 0], solver.p[:, 1], *solver.uh.geo_params)
    if solver.element in ("triangle triangle", "lt"):
        triangles = solver.tri
    else:
        # convert quadrilaterals to triangles
        triangles = _quadrilaterals_to_triangles(solver.tri)

    plt.tricontourf(p_non_ref[:, 0] + solver.uh.x, p_non_ref[:, 1] + solver.uh.y, triangles, solver.uh.von_mises,
                    extend='both', levels=levels, cmap=plt.cm.get_cmap("jet"))
    plt.colorbar()
    # plt.grid()
    xb, xt = np.min(p_non_ref[:, 0] + solver.uh.x), np.max(p_non_ref[:, 0] + solver.uh.x)
    yb, yt = np.min(p_non_ref[:, 1] + solver.uh.y), np.max(p_non_ref[:, 1] + solver.uh.y)
    delta_x = (xt - xb) / 100
    delta_y = (yt - yb) / 100
    plt.xlim(xb - delta_x, xt + delta_x)
    plt.ylim(yb - delta_y, yt + delta_y)
    print("Please call plt.show() to show the plot.")


def plot_rb_von_mises(solver: BaseSolver, levels: Optional[np.ndarray] = None):
    fontsize = 20
    new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
                  'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
                  'figure.titlesize': fontsize,
                  'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
    plt.rcParams.update(new_params)
    if levels is None:
        levels = np.linspace(0, np.max(solver.uh_rom.von_mises), 25)
    plt.figure("RB von mises")
    p_non_ref = solver.vectorized_phi(solver.p[:, 0], solver.p[:, 1], *solver.uh_rom.geo_params)
    if solver.element in ("triangle triangle", "lt"):
        triangles = solver.tri
    else:
        # convert quadrilaterals to triangles
        triangles = _quadrilaterals_to_triangles(solver.tri)

    plt.tricontourf(p_non_ref[:, 0] + solver.uh_rom.x, p_non_ref[:, 1] + solver.uh_rom.y, triangles,
                    solver.uh_rom.von_mises, extend='both', levels=levels, cmap=plt.cm.get_cmap("jet"))
    plt.colorbar()
    # plt.grid()
    xb, xt = np.min(p_non_ref[:, 0] + solver.uh_rom.x), np.max(p_non_ref[:, 0] + solver.uh_rom.x)
    yb, yt = np.min(p_non_ref[:, 1] + solver.uh_rom.y), np.max(p_non_ref[:, 1] + solver.uh_rom.y)
    delta_x = (xt - xb) / 100
    delta_y = (yt - yb) / 100
    plt.xlim(xb - delta_x, xt + delta_x)
    plt.ylim(yb - delta_y, yt + delta_y)
    print("Please call plt.show() to show the plot.")
