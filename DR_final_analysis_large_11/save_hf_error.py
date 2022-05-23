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

from fem_quadrilateral import DraggableCornerRectangleSolver
from fem_quadrilateral import helpers
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp

"""for nice representation of plots"""

fontsize = 20
new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
              'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize, 'figure.titlesize': fontsize,
              'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
plt.rcParams.update(new_params)


def error_saver(root, st_main_root, n):
    d = DraggableCornerRectangleSolver.from_root(root)
    d.matrix_lsq_setup()
    # d.matrix_lsq()
    # d.build_rb_model()
    # get mean snapshot data
    root_mean = root / "mean"
    mean_snapshot = Snapshot(root_mean)
    geo_gird, material_grid, num_geo_param = mean_snapshot["grid_params"]
    geo_range, e_young_range, nu_poisson_range = mean_snapshot["ranges"]
    mode = mean_snapshot["mode_and_element"][0]

    geo_vec = helpers.get_vec_from_range(geo_range, geo_gird, mode)
    geo_mat = np.array(list(product(*repeat(geo_vec, 2))))
    e_young_vec = helpers.get_vec_from_range(e_young_range, material_grid, mode)
    nu_poisson_vec = helpers.get_vec_from_range(nu_poisson_range, material_grid, mode)

    hf_errors = np.zeros(material_grid ** 2)
    for i, (e_young, nu_poisson) in tqdm(enumerate(product(e_young_vec, nu_poisson_vec)), desc="Computing errors"):
        # note n-1 here...
        hf_errors[i] = d.hferror(e_young, nu_poisson, *geo_mat[n - 1, :])
    storage = DiskStorage(st_main_root)
    save_root = storage.root_of(n - 1)
    save_root.mkdir(parents=True, exist_ok=True)
    Snapshot(save_root, hf_errors=hf_errors)
    print(f"saved in {save_root} for n={n}")


def save_hf_errors(p_order, power_divider=3):
    main_root = Path("DR_mls_order_analysis")
    root = main_root / f"p_order_{p_order}"

    st_main_root = Path(f"hf_error_{p_order}")

    print(root)
    print(st_main_root)
    storage = DiskStorage(st_main_root)
    min_n = len(storage)
    print(min_n)

    # must be done to get n_rom_max
    root_mean = root / "mean"
    mean_snapshot = Snapshot(root_mean)
    a_mean = mean_snapshot["a"]
    n_free = a_mean.shape[0]
    ns = 25 ** 2 * 11 ** 2
    print(f"ns: {ns}, n_free: {n_free}, ns <= n_free: {ns <= n_free}.")

    n_max = 25**2
    print(f"n_max={n_max}")
    # only use "1/power_divider power"
    num = max(mp.cpu_count() // power_divider, 1)
    it, r = divmod(n_max - min_n, num)
    for k in range(it):
        min_k = min_n + num * k + 1
        max_k = min_k + num
        pool = mp.Pool(num, maxtasksperchild=1)
        for n in tqdm(range(min_k, max_k), desc=f"Saving batch {k+1} of {it+1}"):
            print(n, "loop 1")
            pool.apply_async(error_saver, [root, st_main_root, n])
        pool.close()
        pool.join()
        del pool

    pool = mp.Pool(num, maxtasksperchild=1)
    for n in tqdm(range(min_n + it * num + 1, min_n + it * num + r + 1), desc=f"Saving batch {it} of {it+1}"):
        print(n, "loop 2")
        pool.apply_async(error_saver, [root, st_main_root, n])
    pool.close()
    pool.join()
    del pool


def load_errors(p_order):
    main_root = Path("DR_mls_order_analysis")
    root = main_root / f"p_order_{p_order}"
    root_mean = root / "mean"
    mean_snapshot = Snapshot(root_mean)
    geo_gird, material_grid, num_geo_param = mean_snapshot["grid_params"]
    geo_range, e_young_range, nu_poisson_range = mean_snapshot["ranges"]
    mode = mean_snapshot["mode_and_element"][0]
    geo_vec = helpers.get_vec_from_range(geo_range, geo_gird, mode)
    e_young_vec = helpers.get_vec_from_range(e_young_range, material_grid, mode)
    nu_poisson_vec = helpers.get_vec_from_range(nu_poisson_range, material_grid, mode)
    param_mat = np.array(list(product(*repeat(geo_vec, num_geo_param), e_young_vec, nu_poisson_vec)))
    error_root = Path(f"hf_error_{p_order}")
    print(error_root)
    errors = np.zeros((25**2, 11**2))
    for i, snapshot in enumerate(DiskStorage(error_root)):
        errors[i, :] = snapshot["hf_errors"]
    errors = errors.ravel()
    print(f"max error: {np.max(errors)} at {param_mat[np.argmax(errors), :]}")
    print(f"mean error: {np.mean(errors)}")
    print(f"min error: {np.min(errors)} at {param_mat[np.argmin(errors), :]}")

if __name__ == '__main__':
    print(datetime.now().time())
    p_order = 19
    power_divider = 3
    save_hf_errors(p_order, power_divider=power_divider)
    print(datetime.now().time())
    load_errors(p_order)



