# -*- coding: utf-8 -*-
"""
Created on 04.04.2022

@author: Olav Milian
"""
from itertools import product, repeat
from pathlib import Path

import numpy as np
from matrix_lsq import Snapshot, DiskStorage

from fem_quadrilateral import DraggableCornerRectangleSolver
from fem_quadrilateral import helpers
from datetime import datetime
from tqdm import tqdm


def save_errors(p_order):
    print("-" * 50)
    main_root = Path("DR_mls_order_analysis")
    root = main_root / f"p_order_{p_order}"
    print(root)
    error_root = Path(f"hf_error_{p_order}")
    print(error_root)
    error_root.mkdir(parents=True, exist_ok=True)

    d = DraggableCornerRectangleSolver.from_root(root)
    d.matrix_lsq_setup()
    d.matrix_lsq()
    print("n_free:", d.n_free)
    root_mean = root / "mean"
    mean_snapshot = Snapshot(root_mean)
    geo_gird, material_grid, num_geo_param = mean_snapshot["grid_params"]
    geo_range, e_young_range, nu_poisson_range = mean_snapshot["ranges"]
    mode = mean_snapshot["mode_and_element"][0]
    geo_vec = helpers.get_vec_from_range(geo_range, geo_gird, mode)
    e_young_vec = helpers.get_vec_from_range(e_young_range, material_grid, mode)
    nu_poisson_vec = helpers.get_vec_from_range(nu_poisson_range, material_grid, mode)
    print("ns:", geo_gird ** num_geo_param * material_grid ** 2)
    if not (error_root / "obj-hf_errors.npy").exists():
        errors_p = np.zeros(geo_gird ** num_geo_param * material_grid ** 2)
        for i, (*geo_params, e_young, nu_poisson) in tqdm(enumerate(
                product(*repeat(geo_vec, num_geo_param), e_young_vec, nu_poisson_vec)), desc="Computing errors"):
            errors_p[i] = d.hferror(e_young, nu_poisson, *geo_params)

        Snapshot(error_root, hf_errors=errors_p)


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
    errors = Snapshot(error_root)["hf_errors"]

    print(f"max error: {np.max(errors)} at {param_mat[np.argmax(errors), :]}")
    print(f"mean error: {np.mean(errors)}")
    print(f"min error: {np.min(errors)} at {param_mat[np.argmin(errors), :]}")


if __name__ == '__main__':
    print(datetime.now().time())
    p_order = 19
    save_errors(p_order)
    print(datetime.now().time())
    load_errors(p_order)

