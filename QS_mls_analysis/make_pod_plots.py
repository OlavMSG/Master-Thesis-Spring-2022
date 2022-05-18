# -*- coding: utf-8 -*-
"""
Created on 04.04.2022

@author: Olav Milian
"""
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from itertools import product, repeat
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matrix_lsq import Snapshot

from src.fem_quadrilateral import QuadrilateralSolver
from src.fem_quadrilateral import helpers
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp

import sympy as sym

"""for nice representation of plots"""

sym.init_printing()
fontsize = 20
new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
              'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize, 'figure.titlesize': fontsize,
              'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
plt.rcParams.update(new_params)


def make_pod_plots(p_order):
    main_root = Path("QS_mls_order_analysis")

    root = main_root / f"p_order_{p_order}"
    save_dict = "plots_pod"
    Path(save_dict).mkdir(parents=True, exist_ok=True)
    print(root)
    d = QuadrilateralSolver.from_root(root)
    d.matrix_lsq_setup()
    d.matrix_lsq(root)
    d.build_rb_model(root)

    d.plot_pod_singular_values()
    plt.savefig("".join((save_dict, f"\\pod_singular_values_p_order_{p_order}.pdf")), bbox_inches='tight')
    plt.show()

    d.plot_pod_relative_information_content()
    plt.ylim(0.999_6, 1.000_05)
    plt.savefig("".join((save_dict, f"\\pod_rel_info_cont_p_order_{p_order}.pdf")), bbox_inches='tight')
    plt.show()

    n_rom_max = d.n_rom_max
    if n_rom_max > 25:
        n_rom_max = 25
    max_errors = np.zeros(n_rom_max)
    mean_errors = np.zeros(n_rom_max)
    min_errors = np.zeros(n_rom_max)

    root_mean = root / "mean"
    mean_snapshot = Snapshot(root_mean)
    geo_gird, material_grid, num_geo_param = mean_snapshot["grid_params"]
    geo_range, e_young_range, nu_poisson_range = mean_snapshot["ranges"]
    mode = mean_snapshot["mode_and_element"][0]

    geo_vec = helpers.get_vec_from_range(geo_range, geo_gird, mode)
    e_young_vec = helpers.get_vec_from_range(e_young_range, material_grid, mode)
    nu_poisson_vec = helpers.get_vec_from_range(nu_poisson_range, material_grid, mode)

    for k, n_rom in enumerate(range(1, n_rom_max + 1)):
        errors_p = np.zeros(geo_gird ** num_geo_param * material_grid ** 2)
        for i, (*geo_params, e_young, nu_poisson) in tqdm(enumerate(
                product(*repeat(geo_vec, num_geo_param), e_young_vec, nu_poisson_vec)), desc="Computing errors"):
            errors_p[i] = d.rberror(root, e_young, nu_poisson, *geo_params, n_rom=n_rom)
        max_errors[k] = np.max(errors_p)
        mean_errors[k] = np.mean(errors_p)
        min_errors[k] = np.min(errors_p)
    print("plotting")
    x = np.arange(n_rom_max) + 1
    plt.figure("err-1")
    plt.title("Relative Errors, $\\|\\|u_h(\\mu)-Vu_N(\\mu)\\|\\|_a/\\|u_h(\\mu)\\|\\|_a$")
    plt.semilogy(x, max_errors, "x--", label="max")
    plt.semilogy(x, mean_errors, "x--", label="mean")
    plt.semilogy(x, min_errors, "x--", label="min")
    plt.xlabel("$N$")
    plt.grid()
    plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
    plt.savefig("".join((save_dict, f"\\pod_errors_p_order_{p_order}.pdf")), bbox_inches='tight')
    plt.show()


def main():
    print(datetime.now().time())
    max_order = 3
    p_order_list = [1, 3]
    multiprocess = False
    if multiprocess:
        pool = mp.Pool(int(min(mp.cpu_count() / 3, len(p_order_list))), maxtasksperchild=1)
        for p_order in p_order_list:
            pool.apply_async(make_pod_plots, [p_order])

        # now we are done,
        pool.close()
        pool.join()
    else:
        for p_order in p_order_list:
            make_pod_plots(p_order)
    print(datetime.now().time())


if __name__ == '__main__':
    main()
