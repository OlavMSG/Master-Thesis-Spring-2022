# -*- coding: utf-8 -*-
"""
Created on 04.04.2022

@author: Olav Milian
"""
from itertools import product, repeat
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matrix_lsq import Snapshot

from fem_quadrilateral import DraggableCornerRectangleSolver
from fem_quadrilateral import helpers
from datetime import datetime
from tqdm import tqdm

import sympy as sym

"""for nice representation of plots"""

sym.init_printing()
fontsize = 20
new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
              'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize, 'figure.titlesize': fontsize,
              'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
plt.rcParams.update(new_params)


def main():
    max_order = 7
    print(datetime.now().time())
    print("-" * 50)
    main_root = Path("DR_mls_order_analysis")
    max_errors = np.zeros(max_order)
    mean_errors = np.zeros(max_order)
    min_errors = np.zeros(max_order)
    for k, p_order in enumerate(range(1, max_order + 1)):
        root = main_root / f"p_order_{p_order}"
        print(root)
        d = DraggableCornerRectangleSolver.from_root(root)
        d.matrix_lsq_setup()
        d.matrix_lsq()

        root_mean = root / "mean"
        mean_snapshot = Snapshot(root_mean)
        geo_gird, material_grid, num_geo_param = mean_snapshot["grid_params"]
        geo_range, e_young_range, nu_poisson_range = mean_snapshot["ranges"]
        mode = mean_snapshot["mode_and_element"][0]
        geo_vec = helpers.get_vec_from_range(geo_range, geo_gird, mode)
        e_young_vec = helpers.get_vec_from_range(e_young_range, material_grid, mode)
        nu_poisson_vec = helpers.get_vec_from_range(nu_poisson_range, material_grid, mode)

        errors_p = np.zeros(geo_gird ** num_geo_param * material_grid ** 2)
        for i, (*geo_params, e_young, nu_poisson) in tqdm(enumerate(
                product(*repeat(geo_vec, num_geo_param), e_young_vec, nu_poisson_vec)), desc="Computing errors"):
            errors_p[i] = d.hferror(e_young, nu_poisson, *geo_params)
        max_errors[k] = np.max(errors_p)
        mean_errors[k] = np.mean(errors_p)
        min_errors[k] = np.min(errors_p)
    print("plotting")
    save_dict = "plots_DR_mls_order_analysis"
    Path(save_dict).mkdir(parents=True, exist_ok=True)
    x = np.arange(max_order) + 1
    plt.figure("err-1")
    plt.title("Relative Errors, $\\|\\|u_h(\\mu)-u_{h,mls}(\\mu)\\|\\|_a/\\|\\|u_h(\\mu)\\|\\|_a$")
    plt.semilogy(x, max_errors, "x--", label="max")
    plt.semilogy(x, mean_errors, "x--", label="mean")
    plt.semilogy(x, min_errors, "x--", label="min")
    plt.xlabel("$p$, order")
    plt.grid()
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
    plt.savefig("".join((save_dict, "\\relative_errors.pdf")), bbox_inches='tight')
    plt.show()
    print(datetime.now().time())


if __name__ == '__main__':
    main()
