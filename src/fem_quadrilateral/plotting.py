# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from .base_solver import BaseSolver


def plot_mesh(solver: BaseSolver, *geo_params: float):
    # set nice plotting
    fontsize = 20
    new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
                  'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
                  'figure.titlesize': fontsize,
                  'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
    plt.rcParams.update(new_params)
    plt.figure("Mesh plot", figsize=(7, 7))
    plt.title(f"Mesh {solver.n}x{solver.n}")
    p_non_ref = solver.vectorized_phi(solver.p[:, 0], solver.p[:, 1], *geo_params)

    if solver.element in ("triangle triangle", "lt"):
        plt.triplot(p_non_ref[:, 0], p_non_ref[:, 1], solver.tri)
    else:
        verts = p_non_ref[solver.tri]
        pc = PolyCollection(verts, color="blue", linewidths=2, facecolor="None")
        ax = plt.gca()
        ax.add_collection(pc)
        ax.autoscale()
    plt.grid()
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
    plt.title(f"Mesh {solver.n}x{solver.n}")

    pod_mode = solver.rb_pod_mode(i)
    p_non_ref = solver.vectorized_phi(solver.p[:, 0], solver.p[:, 1], *pod_mode.geo_params)
    if solver.element in ("triangle triangle", "lt"):
        plt.triplot(p_non_ref[:, 0] + pod_mode.x, p_non_ref[:, 1] + pod_mode.y, solver.tri)
    else:
        verts = (p_non_ref + pod_mode.values)[solver.tri]
        pc = PolyCollection(verts, color="blue", linewidths=2, facecolor="None")
        ax = plt.gca()
        ax.add_collection(pc)
        ax.autoscale()
    plt.grid()
    xb, xt = plt.xlim()
    yb, yt = plt.ylim()
    xy_min = min(xb, yb)
    xy_max = max(xt, yt)
    plt.xlim(xy_min, xy_max)
    plt.ylim(xy_min, xy_max)
    print("Please call plt.show() to show the plot.")


def plot_hf_displacment(solver: BaseSolver):
    # set nice plotting
    fontsize = 20
    new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (7, 7),
                  'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
                  'figure.titlesize': fontsize,
                  'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
    plt.rcParams.update(new_params)
    title_text = f"Displacement in high-fidelity solution, $n={solver.n}$"
    plt.figure(title_text)
    plt.title(title_text)
    colors1 = np.ones(solver.tri.shape[0])

    p_non_ref = solver.vectorized_phi(solver.p[:, 0], solver.p[:, 1], *solver.uh.geo_params)
    if solver.element in ("triangle triangle", "lt"):
        cmap1 = colors.ListedColormap("red")
        cmap2 = colors.ListedColormap("gray")
        plt.tripcolor(p_non_ref[:, 0] + solver.uh.x, p_non_ref[:, 1] + solver.uh.y,
                      solver.tri, facecolors=colors1, cmap=cmap1)
        plt.tripcolor(p_non_ref[:, 0], p_non_ref[:, 1],
                      solver.tri, facecolors=colors1, cmap=cmap2, alpha=0.5)
    else:
        ax = plt.gca()
        verts1 = (p_non_ref + solver.uh.values)[solver.tri]
        pc1 = PolyCollection(verts1, linewidths=2, facecolors=colors1, color="red")
        ax.add_collection(pc1)
        verts2 = p_non_ref[solver.tri]
        pc2 = PolyCollection(verts2, linewidths=2, facecolor=colors1, color="gray", alpha=0.5)
        ax.add_collection(pc2)
        ax.autoscale()
    plt.grid()
    xb, xt = plt.xlim()
    yb, yt = plt.ylim()
    xy_min = min(xb, yb)
    xy_max = max(xt, yt)
    plt.xlim(xy_min, xy_max)
    plt.ylim(xy_min, xy_max)
    print("Please call plt.show() to show the plot.")


def plot_rb_displacment(solver: BaseSolver):
    # set nice plotting
    fontsize = 20
    new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
                  'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
                  'figure.titlesize': fontsize,
                  'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
    plt.rcParams.update(new_params)
    title_text = f"Displacement in reduced-order solution, $n={solver.n}$"
    plt.figure(title_text)
    plt.title(title_text)
    colors1 = np.ones(solver.tri.shape[0])

    p_non_ref = solver.vectorized_phi(solver.p[:, 0], solver.p[:, 1], *solver.uh_rom.geo_params)
    if solver.element in ("triangle triangle", "lt"):
        cmap1 = colors.ListedColormap("red")
        cmap2 = colors.ListedColormap("gray")
        plt.tripcolor(p_non_ref[:, 0] + solver.uh_rom.x, p_non_ref[:, 1] + solver.uh_rom.y,
                      solver.tri, facecolors=colors1, cmap=cmap1)
        plt.tripcolor(p_non_ref[:, 0], p_non_ref[:, 1],
                      solver.tri, facecolors=colors1, cmap=cmap2, alpha=0.5)
    else:
        ax = plt.gca()
        verts1 = (p_non_ref + solver.uh_rom.values)[solver.tri]
        pc1 = PolyCollection(verts1, linewidths=2, facecolors=colors1, color="red")
        ax.add_collection(pc1)
        verts2 = p_non_ref[solver.tri]
        pc2 = PolyCollection(verts2, linewidths=2, facecolor=colors1, color="gray", alpha=0.5)
        ax.add_collection(pc2)
        ax.autoscale()
    plt.grid()
    xb, xt = plt.xlim()
    yb, yt = plt.ylim()
    xy_min = min(xb, yb)
    xy_max = max(xt, yt)
    plt.xlim(xy_min, xy_max)
    plt.ylim(xy_min, xy_max)
    print("Please call plt.show() to show the plot.")


def _quadrilaterals_to_triangles(quads: np.ndarray) -> np.ndarray:
    tris = np.zeros((2 * len(quads), 3))
    for i in range(len(quads)):
        j = 2 * i
        tris[j, :] = quads[i][[0, 1, 2]]
        tris[j + 1, :] = quads[i][[0, 3, 2]]
    return tris


def plot_hf_von_mises(solver: BaseSolver):
    levels = np.linspace(0, np.max(solver.uh.von_mises), 25)
    title_text = f"Von Mises stress in high-fidelity solution, $n={solver.n}$"
    plt.figure(title_text)
    plt.title(title_text)
    p_non_ref = solver.vectorized_phi(solver.p[:, 0], solver.p[:, 1], *solver.uh.geo_params)
    if solver.element in ("triangle triangle", "lt"):
        triangles = solver.tri
    else:
        # convert quadrilaterals to triangles
        triangles = _quadrilaterals_to_triangles(solver.tri)
    plt.gca().set_aspect('equal')
    plt.tricontourf(p_non_ref[:, 0] + solver.uh.x, p_non_ref[:, 1] + solver.uh.y, triangles, solver.uh.von_mises,
                    extend='both', levels=levels, cmap=plt.cm.get_cmap("jet"))
    plt.colorbar()
    plt.grid()
    plt.xlim(np.min(p_non_ref[:, 0] + solver.uh.x) - 0.05, np.max(p_non_ref[:, 0] + solver.uh.x) + 0.05)
    plt.ylim(np.min(p_non_ref[:, 1] + solver.uh.y) - 0.05, np.max(p_non_ref[:, 1] + solver.uh.y) + 0.05)
    print("Please call plt.show() to show the plot.")


def plot_rb_von_mises(solver: BaseSolver):
    levels = np.linspace(0, np.max(solver.uh_rom.von_mises), 25)
    title_text = f"Von Mises stress in high-fidelity solution, $n={solver.n}$"
    plt.figure(title_text)
    plt.title(title_text)
    p_non_ref = solver.vectorized_phi(solver.p[:, 0], solver.p[:, 1], *solver.uh_rom.geo_params)
    if solver.element in ("triangle triangle", "lt"):
        triangles = solver.tri
    else:
        # convert quadrilaterals to triangles
        triangles = _quadrilaterals_to_triangles(solver.tri)
    plt.gca().set_aspect('equal')
    plt.tricontourf(p_non_ref[:, 0] + solver.uh_rom.x, p_non_ref[:, 1] + solver.uh_rom.y, triangles,
                    solver.uh_rom.von_mises, extend='both', levels=levels, cmap=plt.cm.get_cmap("jet"))
    plt.colorbar()
    plt.grid()
    plt.xlim(np.min(p_non_ref[:, 0] + solver.uh_rom.x) - 0.05, np.max(p_non_ref[:, 0] + solver.uh_rom.x) + 0.05)
    plt.ylim(np.min(p_non_ref[:, 1] + solver.uh_rom.y) - 0.05, np.max(p_non_ref[:, 1] + solver.uh_rom.y) + 0.05)
    print("Please call plt.show() to show the plot.")
