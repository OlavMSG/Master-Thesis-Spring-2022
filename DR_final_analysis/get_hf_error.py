# -*- coding: utf-8 -*-
"""
Created on 04.04.2022

@author: Olav Milian
"""
from itertools import product, repeat
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matrix_lsq import Snapshot

from src.fem_quadrilateral import DraggableCornerRectangleSolver
from src.fem_quadrilateral import helpers
from datetime import datetime
from tqdm import tqdm

import sympy as sym


def main():
    p_order = 19
    print(datetime.now().time())
    print("-" * 50)
    main_root = Path("DR_mls_order_analysis")
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
    print(f"max error: {np.max(errors_p)}")
    print(f"mean error: {np.mean(errors_p)}")
    print(f"min error: {np.min(errors_p)}")
    print(datetime.now().time())


if __name__ == '__main__':
    main()
