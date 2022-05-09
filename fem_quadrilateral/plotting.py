# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from .base_solver import BaseSolver


def plot_mesh(solver: BaseSolver, *geo_params: float, element="bq"):
    # set nice plotting
    fontsize = 20
    new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
                  'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
                  'figure.titlesize': fontsize,
                  'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
    plt.rcParams.update(new_params)
    plt.figure("Mesh plot", figsize=(7, 7))
    plt.title(f"Mesh {solver.n}x{solver.n}")
    p_non_ref = solver.phi(solver.p[:, 0], solver.p[:, 1], *geo_params)

    if element in ("linear triangle", "lt"):
        plt.triplot(p_non_ref[:, 0], p_non_ref[:, 1], solver.tri)
    else:
        verts = p_non_ref[solver.tri]
        pc = PolyCollection(verts, color="blue", linewidths=2, facecolor="None")
        ax = plt.gca()
        ax.add_collection(pc)
        ax.autoscale()
    plt.grid()
    plt.show()

def plot_hf_displacment(solver: BaseSolver, *geo_params: float, element="bq"):
    # set nice plotting
    fontsize = 20
    new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
                  'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
                  'figure.titlesize': fontsize,
                  'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
    plt.rcParams.update(new_params)
    title_text = f"Displacement in high-fidelity solution, $n={solver.n}$"
    plt.figure(title_text)
    plt.title(title_text)
    colors1 = np.ones(solver.tri.shape[0])
    cmap1 = colors.ListedColormap("red")
    cmap2 = colors.ListedColormap("gray")

    p_non_ref = solver.phi(solver.p[:, 0], solver.p[:, 1], *geo_params)
    if element in ("linear triangle", "lt"):
        plt.tripcolor(p_non_ref[:, 0] + solver.uh.x, p_non_ref[:, 1] + solver.uh.y,
                      solver.tri, facecolors=colors1, cmap=cmap1)
        plt.tripcolor(p_non_ref[:, 0], p_non_ref[:, 1],
                      solver.tri, facecolors=colors1, cmap=cmap2, alpha=0.5)
    else:
        ax = plt.gca()
        verts1 = (p_non_ref + solver.uh.values)[solver.tri]
        pc1 = PolyCollection(verts1, linewidths=2, facecolors=colors1, cmap=cmap1)
        ax.add_collection(pc1)
        verts2 = p_non_ref[solver.tri]
        pc2 = PolyCollection(verts2, linewidths=2, facecolor=colors1, cmap=cmap2, alpha=0.5)
        ax.add_collection(pc2)
        ax.autoscale()
    plt.grid()

def plot_rb_displacment(solver: BaseSolver, *geo_params: float, element="bq"):
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
    cmap1 = colors.ListedColormap("red")
    cmap2 = colors.ListedColormap("gray")

    p_non_ref = solver.phi(solver.p[:, 0], solver.p[:, 1], *geo_params)
    if element in ("linear triangle", "lt"):
        plt.tripcolor(p_non_ref[:, 0] + solver.uh_rom.x, p_non_ref[:, 1] + solver.uh_rom.y,
                      solver.tri, facecolors=colors1, cmap=cmap1)
        plt.tripcolor(p_non_ref[:, 0], p_non_ref[:, 1],
                      solver.tri, facecolors=colors1, cmap=cmap2, alpha=0.5)
    else:
        ax = plt.gca()
        verts1 = (p_non_ref + solver.uh_rom.values)[solver.tri]
        pc1 = PolyCollection(verts1, linewidths=2, facecolors=colors1, cmap=cmap1)
        ax.add_collection(pc1)
        verts2 = p_non_ref[solver.tri]
        pc2 = PolyCollection(verts2, linewidths=2, facecolor=colors1, cmap=cmap2, alpha=0.5)
        ax.add_collection(pc2)
        ax.autoscale()
    plt.grid()

