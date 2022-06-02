# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from itertools import product
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import sys
import timeit

from matrix_lsq import DiskStorage

from fem_quadrilateral import DraggableCornerRectangleSolver, default_constants, helpers
from fem_quadrilateral.default_constants import e_young_range, nu_poisson_range

# rho_steal = 8e3  # kg/m^3
alpha = 8e3 * 100 * 9.81 * 0.01  # N/m^2

e_mean = np.mean(e_young_range)
nu_mean = np.mean(nu_poisson_range)
mu1, mu2 = 0.2, -0.2


# Example 2: Gravity in 2D
def f(x, y):
    return alpha, 0


def clamped_bc(x, y):
    return abs(x) <= default_constants.default_tol


class SaveOneSnapshotSimulator:
    storage: DiskStorage
    root: Path
    geo_grid: int
    geo_range: Tuple[float, float]
    mode: str
    material_grid: int = default_constants.e_nu_grid
    e_young_range: Tuple[float, float] = default_constants.e_young_range
    nu_poisson_range: Tuple[float, float] = default_constants.nu_poisson_range

    def __init__(self, root: Path,
                 mode: str = "uniform",
                 material_grid: Optional[int] = None,
                 e_young_range: Optional[Tuple[float, float]] = None,
                 nu_poisson_range: Optional[Tuple[float, float]] = None):
        self.root = root
        self.storage = DiskStorage(root)
        self.mode = mode

        if material_grid is not None:
            self.material_grid = material_grid
        if e_young_range is not None:
            self.e_young_range = e_young_range
        if nu_poisson_range is not None:
            self.nu_poisson_range = nu_poisson_range

        assert len(self.storage) == 0

    def __call__(self, solver: DraggableCornerRectangleSolver, *geo_params):
        geo_vec = helpers.get_vec_from_range(DraggableCornerRectangleSolver.geo_param_range, 25, self.mode)
        e_young_vec = helpers.get_vec_from_range(self.e_young_range, self.material_grid, self.mode)
        nu_poisson_vec = helpers.get_vec_from_range(self.nu_poisson_range, self.material_grid, self.mode)
        # make snapshots
        solver.assemble(*geo_params)
        # compute solution and a-norm-squared for all e_young and nu_poisson
        # put in solution matrix and anorm2 vector
        s_mat = np.zeros((solver.n_free, self.material_grid ** 2))
        uh_anorm2_vec = np.zeros(self.material_grid ** 2)
        for i, (e_young, nu_poisson) in enumerate(product(e_young_vec, nu_poisson_vec)):
            solver.hfsolve(e_young, nu_poisson, print_info=False)
            s_mat[:, i] = solver.uh_free
            uh_anorm2_vec[i] = solver.uh_anorm2
        # matrix-LSQ data
        data = solver.mls_funcs(*geo_params).ravel()
        # save
        if solver.has_non_homo_dirichlet:
            self.storage.append(data, a1=solver.a1, a2=solver.a2,
                                f0=solver.f0,
                                f1_dir=solver.f1_dir, f2_dir=solver.f2_dir,
                                s_mat=s_mat, uh_anorm2=uh_anorm2_vec)

        else:
            self.storage.append(data, a1=solver.a1, a2=solver.a2,
                                f0=solver.f0,
                                s_mat=s_mat, uh_anorm2=uh_anorm2_vec)


if __name__ == '__main__':
    n = 90
    # assuming that n = 90 takes approx. 9 hours
    with open(f"time_code_log_n{n}.txt", "w") as time_code_log:
        sys.stdout = time_code_log
        p_order = 19
        main_root = Path("DR_mls_order_analysis")
        root = main_root / f"p_order_{p_order}"
        print(f"root: {root}")
        # Degrees of freedom info
        rec = DraggableCornerRectangleSolver.from_root(root)
        print("Degrees of freedom info")
        print("-" * 50)
        print(f"Nodes along one axis n: {rec.n}")
        print(f"HF system full size n_full: {rec.n_full}")
        print(f"ns for solution matrix (ns_rom): {rec.ns_rom}")
        print("-" * 50)
        print(f"HF dofs Nh (n_free): {rec.n_free}")
        print(f"RB dofs N (n_rom): {rec.n_rom}")
        print(f"Dofs reduction: {round(rec.n_free / rec.n_rom)} to 1, ({rec.n_free / rec.n_rom} to 1)")
        print("-" * 50)

        # Assemble HF system, do not use saved data
        num1 = 50  # x "2 min" = 1 hour 40 min
        setup = "d = DraggableCornerRectangleSolver(n, f_func=f, get_dirichlet_edge_func=clamped_bc, " \
                "bcs_are_on_reference_domain=False)"
        code = "d.assemble(mu1, mu2)"
        time1 = timeit.timeit(code, number=num1, globals=globals(), setup=setup)
        print("Assemble HF system:")
        print(f"total : {time1}  sec, mean time: {time1 / num1} sec, runs: {num1}")
        print("-" * 50)

        # save one snapshot
        timeing_root = main_root / f"timing_{p_order}"
        num11 = 50  # x "2 min" = 1 hour 40 min
        setup = "d = DraggableCornerRectangleSolver(n, f_func=f, get_dirichlet_edge_func=clamped_bc, " \
                "bcs_are_on_reference_domain=False); d.matrix_lsq_setup(p_order); " \
                "s = SaveOneSnapshotSimulator(timeing_root)"
        code = "s(d, mu1, mu2)"
        time11 = timeit.timeit(code, number=num1, globals=globals(), setup=setup)
        print("Save One Snapshot:")
        print(f"total : {time11}  sec, mean time: {time11 / num11} sec, runs: {num11}")
        print("-" * 50)

        # Solve HF system, do not use saved data
        num2 = 1_000  # x "0.9s" = 15 min
        setup = "d = DraggableCornerRectangleSolver(n, f_func=f, get_dirichlet_edge_func=clamped_bc, " \
                "bcs_are_on_reference_domain=False); d.assemble(mu1, mu2)"
        code = "d.hfsolve(e_mean, nu_mean)"
        time2 = timeit.timeit(code, number=num2, globals=globals(), setup=setup)
        print("Solve HF system:")
        print(f"total : {time2} sec, mean time: {time2 / num2} sec, runs: {num2}")
        print("-" * 50)

        # Matrix Least Square, use saved data (data has MLS and RB saved)
        num21 = 10  # x "9 min" = 1 hours 30 min
        setup = "d = DraggableCornerRectangleSolver.from_root(root); d.matrix_lsq_setup(p_order)"
        code = "d.matrix_lsq(root)"
        time21 = timeit.timeit(code, number=num21, globals=globals(), setup=setup)
        print("Matrix Least Square")
        print(f"total : {time21} sec, mean time: {time21 / num21} sec, runs: {num21}")
        print("-" * 50)

        # Build RB system, use saved data (data has MLS and RB saved)
        num3 = 10  # x "24 min" = 4 hours
        setup = "d = DraggableCornerRectangleSolver.from_root(root); d.matrix_lsq_setup(p_order)"
        code = "d.build_rb_model(root)"
        time3 = timeit.timeit(code, number=num3, globals=globals(), setup=setup)
        print("Build RB model:")
        print(f"total : {time3} sec, mean time: {time3 / num3} sec, runs: {num3}")
        print("-" * 50)

        # Solve RB system, do not use saved data
        num4 = 1_000  # x "1.6 ms" = less than 2 s
        setup = "d = DraggableCornerRectangleSolver.from_root(root); d.matrix_lsq_setup(p_order)"
        code = "d.rbsolve_uh_rom_non_recovered(e_mean, nu_mean, mu1, mu2)"
        time4 = timeit.timeit(code, number=num4, globals=globals(), setup=setup)
        print("Solve RB system:")
        print(f"total : {time4} sec, mean time: {time4 / num4} sec, runs: {num4}")
        print("-" * 50)

        # 1 x Assemble HF + 625 x Save Snapshots + MLS + Build RB +
        print(f"Offline CPU time: {time1 / num1 + 625 * time11 / num11 + time21 / num21 + time3 / num3} sec")
        print(f"Online CPU time: {time4 / num4} sec")
