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
from matrix_lsq import Snapshot, DiskStorage

from src.fem_quadrilateral import QuadrilateralSolver
from src.fem_quadrilateral import helpers
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp

"""for nice representation of plots"""

fontsize = 20
new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
              'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize, 'figure.titlesize': fontsize,
              'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
plt.rcParams.update(new_params)


def error_saver(root, st_main_root, n_rom):
    # get mean snapshot data
    root_mean = root / "mean"
    mean_snapshot = Snapshot(root_mean)
    geo_gird, material_grid, num_geo_param = mean_snapshot["grid_params"]
    geo_range, e_young_range, nu_poisson_range = mean_snapshot["ranges"]
    mode = mean_snapshot["mode_and_element"][0]

    geo_vec = helpers.get_vec_from_range(geo_range, geo_gird, mode)
    e_young_vec = helpers.get_vec_from_range(e_young_range, material_grid, mode)
    nu_poisson_vec = helpers.get_vec_from_range(nu_poisson_range, material_grid, mode)

    d = QuadrilateralSolver.from_root(root)
    d.matrix_lsq_setup()
    d.matrix_lsq()
    d.build_rb_model()

    errors_p = np.zeros(geo_gird ** num_geo_param * material_grid ** 2)
    for i, (*geo_params, e_young, nu_poisson) in tqdm(enumerate(
            product(*repeat(geo_vec, num_geo_param), e_young_vec, nu_poisson_vec)), desc="Computing errors"):
        errors_p[i] = d.rberror(e_young, nu_poisson, *geo_params, n_rom=n_rom)
    errors = np.array([np.max(errors_p), np.mean(errors_p), np.min(errors_p)])
    arg_errors = np.array([np.argmax(errors_p), np.argmin(errors_p)])
    storage = DiskStorage(st_main_root)
    save_root = storage.root_of(n_rom - 1)
    save_root.mkdir(parents=True, exist_ok=True)
    Snapshot(save_root, errors=errors, arg_errors=arg_errors)
    print(f"saved in {save_root} using n_rom={n_rom}")


def save_pod_errors(p_order, power_divider=3, set_n_rom_max=False):
    main_root = Path("QS_mls_order_analysis")
    root = main_root / f"p_order_{p_order}"

    st_main_root = Path(f"pod_errors_{p_order}")

    print(root)
    print(st_main_root)
    storage = DiskStorage(st_main_root)
    min_n_rom = len(storage)
    print(min_n_rom)

    # must be done to get n_rom_max
    """d = QuadrilateralSolver.from_root(root)
    d.matrix_lsq_setup()
    d.matrix_lsq(root)
    d.build_rb_model(root)"""

    n_rom_max = 330  # here d.n_rom_max
    print(f"n_rom_max={n_rom_max}")
    if set_n_rom_max:
        n_rom_max = 25
    # only use "1/power_divider power"
    num = max(mp.cpu_count() // power_divider, 1)
    it, r = divmod(n_rom_max - min_n_rom, num)
    for k in range(it):
        min_k = min_n_rom + num * k + 1
        max_k = min_k + num
        pool = mp.Pool(num, maxtasksperchild=1)
        for n_rom in range(min_k, max_k):
            print(n_rom, "loop 1")
            pool.apply_async(error_saver, [root, st_main_root, n_rom])
        pool.close()
        pool.join()
        del pool

    pool = mp.Pool(num, maxtasksperchild=1)
    for n_rom in range(min_n_rom + it * num + 1, min_n_rom + it * num + r + 1):
        print(n_rom, "loop 2")
        pool.apply_async(error_saver, [root, st_main_root, n_rom])
    pool.close()
    pool.join()
    del pool


def plot_pod_errors(p_order):
    main_root = Path("QS_mls_order_analysis")
    root = main_root / f"p_order_{p_order}"

    st_main_root = Path(f"pod_errors_{p_order}")

    print(root)
    print(st_main_root)

    save_dict = "plots_pod"
    Path(save_dict).mkdir(parents=True, exist_ok=True)

    # must be done to get n_rom_max
    d = QuadrilateralSolver.from_root(root)
    d.matrix_lsq_setup()
    d.matrix_lsq()
    d.build_rb_model()

    n_rom_max = d.n_rom_max
    print(n_rom_max)

    d.plot_pod_singular_values()
    plt.savefig("".join((save_dict, f"\\pod_singular_values_p_order_{p_order}.pdf")), bbox_inches='tight')
    plt.show()

    d.plot_pod_relative_information_content()
    plt.ylim(0.999_6, 1.000_05)
    plt.savefig("".join((save_dict, f"\\pod_rel_info_cont_p_order_{p_order}.pdf")), bbox_inches='tight')
    plt.show()

    root_mean = root / "mean"
    mean_snapshot = Snapshot(root_mean)
    geo_gird, material_grid, num_geo_param = mean_snapshot["grid_params"]
    geo_range, e_young_range, nu_poisson_range = mean_snapshot["ranges"]
    mode = mean_snapshot["mode_and_element"][0]

    geo_vec = helpers.get_vec_from_range(geo_range, geo_gird, mode)
    e_young_vec = helpers.get_vec_from_range(e_young_range, material_grid, mode)
    nu_poisson_vec = helpers.get_vec_from_range(nu_poisson_range, material_grid, mode)

    param_mat = np.array(list(product(*repeat(geo_vec, num_geo_param), e_young_vec, nu_poisson_vec)))

    # get errors
    storage = DiskStorage(st_main_root)
    max_errors = np.zeros(len(storage))
    mean_errors = np.zeros(len(storage))
    min_errors = np.zeros(len(storage))
    n_roms = np.zeros((len(storage)))
    print("-"*50)
    for k, snapshot in enumerate(storage):
        n_roms[k] = k + 1
        max_errors[k], mean_errors[k], min_errors[k] = snapshot["errors"]
        arg_errors = snapshot["arg_errors"].astype(int)
        print(f"N={k+1}:\nMax error at {param_mat[arg_errors[0]]}\nMin error at {param_mat[arg_errors[1]]}")
    print(dict(zip(n_roms, mean_errors)))
    print(np.all(n_roms - 1 - np.arange(len(n_roms)) == 0))

    plt.figure("err-1")
    plt.title("Relative Errors, $\\|\\|u_h(\\mu)-Vu_N(\\mu)\\|\\|_a/\\|u_h(\\mu)\\|\\|_a$")
    plt.semilogy(n_roms, max_errors, "x--", label="max")
    plt.semilogy(n_roms, mean_errors, "x--", label="mean")
    plt.semilogy(n_roms, min_errors, "x--", label="min")
    plt.xlabel("$N$")
    plt.grid()
    plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
    plt.savefig("".join((save_dict, f"\\pod_errors_p_order_{p_order}.pdf")), bbox_inches='tight')
    plt.show()


def main():
    print(datetime.now().time())
    p_order = 4
    power_divider = 3
    # save_pod_errors(p_order, power_divider=power_divider)
    print(datetime.now().time())
    plot_pod_errors(p_order)


if __name__ == '__main__':
    main()
