# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from pathlib import Path

import numpy as np
from matrix_lsq import DiskStorage, Snapshot
from scipy.linalg import fractional_matrix_power, eigh


def main():
    main_root = Path("DR_mls_order_analysis")
    root = main_root / f"p_order_{19}_old"
    root_mean = root / "mean"
    mean_snapshot = Snapshot(root_mean)
    a_mean = mean_snapshot["a"]
    n_free = a_mean.shape[0]
    geo_gird, material_grid, num_geo_param = mean_snapshot["grid_params"]
    m = material_grid ** 2
    ns = geo_gird ** num_geo_param * m

    s_mat = np.zeros((n_free, ns), dtype=float)
    import tqdm
    from time import perf_counter

    for i, snapshot in tqdm.tqdm(enumerate(DiskStorage(root)), desc="Loading data"):
        s_mat[:, i * m:(i + 1) * m] = snapshot["s_mat"]
    if (s_mat == 0).all():
        error_text = "Solution matrix is zero, can not compute POD for building a reduced model. " \
                     + "The most likely cause is f_func=0, dirichlet_bc_func=0 and neumann_bc_func=0, " \
                     + "where two last may be None."
        raise ValueError(error_text)
    print(a_mean.shape)
    print(s_mat.shape)
    # build correlation matrix
    corr_mat = s_mat.T @ a_mean @ s_mat
    print(corr_mat.shape)
    print("corr_computed")
    # find the eigenvalues and eigenvectors of it
    s = perf_counter()
    sigma2_vec1, zeta_mat1 = eigh(corr_mat)
    print("time eigh:", perf_counter() - s)
    sigma2_vec1 = sigma2_vec1[::-1]
    zeta_mat1 = zeta_mat1[:, ::-1]

    n_rom_max1 = np.linalg.matrix_rank(s_mat)
    v_mat_n_max1 = s_mat @ zeta_mat1[:, :n_rom_max1] / np.sqrt(sigma2_vec1[:n_rom_max1])

    s = perf_counter()
    x05 = fractional_matrix_power(a_mean.A, 0.5)
    print("time x05:", perf_counter() - s)
    # build matrix
    k_mat = x05 @ s_mat @ s_mat.T @ x05
    print(k_mat.shape)
    # find the eigenvalues and eigenvectors of it

    print("corr_computed")
    # find the eigenvalues and eigenvectors of it
    s = perf_counter()
    sigma2_vec2, zeta_mat2 = eigh(k_mat)
    print("time eigh:", perf_counter() - s)
    sigma2_vec2 = sigma2_vec2[::-1]
    zeta_mat2 = zeta_mat2[:, ::-1]

    n_rom_max2 = np.linalg.matrix_rank(s_mat)
    v_mat_n_max2 = np.linalg.solve(x05, zeta_mat2[:, :n_rom_max2])

    print(sigma2_vec1[:30])
    print(sigma2_vec2[:30])
    n = min(len(sigma2_vec1), len(sigma2_vec2))
    print(np.max(np.abs(sigma2_vec1[:n] - sigma2_vec2[:n])), np.argmax(np.abs(sigma2_vec1[:n] - sigma2_vec2[:n])))
    print(v_mat_n_max1.shape, v_mat_n_max2.shape)
    print(np.max(np.abs(v_mat_n_max1 - v_mat_n_max2)))
    max_e, arg_e = -1, -1
    for i, col in enumerate(np.abs(v_mat_n_max1 - v_mat_n_max2).T):
        pot_max = np.max(col)
        if pot_max > max_e:
            max_e = pot_max
            arg_e = i
    print(arg_e)


if __name__ == '__main__':
    main()
