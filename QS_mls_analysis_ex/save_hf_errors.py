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

from fem_quadrilateral import QuadrilateralSolver, helpers
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp

"""for nice representation of plots"""

fontsize = 20
new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
              'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize, 'figure.titlesize': fontsize,
              'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
plt.rcParams.update(new_params)


def error_saver(root, st_main_root, p_order):
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
    d.matrix_lsq(root)
    d.build_rb_model(root)

    errors_p = np.zeros(geo_gird ** num_geo_param * material_grid ** 2)
    for i, (*geo_params, e_young, nu_poisson) in tqdm(enumerate(
            product(*repeat(geo_vec, num_geo_param), e_young_vec, nu_poisson_vec)), desc="Computing errors"):
        errors_p[i] = d.hferror(root, e_young, nu_poisson, *geo_params)
    errors = np.array([np.max(errors_p), np.mean(errors_p), np.min(errors_p)])
    storage = DiskStorage(st_main_root)
    save_root = storage.root_of(p_order - 1)
    save_root.mkdir(parents=True, exist_ok=True)
    Snapshot(save_root, errors=errors)
    print(f"saved in {save_root} using p_order={p_order}")


def save_pod_errors(max_order, power_divider=3):
    main_root = Path("QS_mls_order_analysis")

    st_main_root = Path(f"hf_errors")

    print(st_main_root)
    storage = DiskStorage(st_main_root)
    min_order = len(storage)
    print(min_order)
    print(max_order)
    # only use "1/power_divider power"
    num = max(mp.cpu_count() // power_divider, 1)
    it, r = divmod(max_order - min_order, num)
    for k in range(it):
        min_k = min_order + num * k + 1
        max_k = min_k + num
        pool = mp.Pool(num, maxtasksperchild=1)
        for p_order in range(min_k, max_k):
            root = main_root / f"p_order_{p_order}"
            print(p_order, "loop 1", root)
            pool.apply_async(error_saver, [root, st_main_root, p_order])
        pool.close()
        pool.join()
        del pool

    pool = mp.Pool(num, maxtasksperchild=1)
    for p_order in range(min_order + it * num + 1, min_order + it * num + r + 1):
        root = main_root / f"p_order_{p_order}"
        print(p_order, "loop 2", root)
        pool.apply_async(error_saver, [root, st_main_root, p_order])
    pool.close()
    pool.join()
    del pool


def plot_hf_errors():

    st_main_root = Path(f"hf_errors")
    print(st_main_root)

    save_dict = "plots_QS_mls_order_analysis"
    Path(save_dict).mkdir(parents=True, exist_ok=True)
    # get errors
    storage = DiskStorage(st_main_root)
    max_errors = np.zeros(len(storage))
    mean_errors = np.zeros(len(storage))
    min_errors = np.zeros(len(storage))
    p_orders = np.zeros((len(storage)))
    print("-"*50)
    for k, snapshot in enumerate(storage):
        p_orders[k] = k + 1
        max_errors[k], mean_errors[k], min_errors[k] = snapshot["errors"]

    print(dict(zip(p_orders, mean_errors)))
    print(np.all(p_orders - 1 - np.arange(len(p_orders)) == 0))

    plt.figure("err-1")
    plt.title("Relative Errors, $\\|\\|u_h(\\mu)-u_{h,mls}(\\mu)\\|\\|_a/\\|u_h(\\mu)\\|\\|_a$")
    plt.semilogy(p_orders, max_errors, "x--", label="max")
    plt.semilogy(p_orders, mean_errors, "x--", label="mean")
    plt.semilogy(p_orders, min_errors, "x--", label="min")
    plt.xlabel("$p$, order")
    plt.grid()
    plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
    plt.savefig("".join((save_dict, "\\relative_errors.pdf")), bbox_inches='tight')
    plt.show()


def main():
    print(datetime.now().time())
    max_order = 5
    power_divider = 3
    # save_pod_errors(max_order, power_divider=power_divider)
    print(datetime.now().time())
    plot_hf_errors()


if __name__ == '__main__':
    main()
