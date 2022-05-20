# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from pathlib import Path

import numpy as np
from matrix_lsq import DiskStorage, Snapshot

from fem_quadrilateral import DraggableCornerRectangleSolver, default_constants
from datetime import datetime


def save_mls_and_rb_model(root, p_order):
    d = DraggableCornerRectangleSolver.from_root(root)
    d.matrix_lsq_setup()
    print("mu_range:", d.geo_param_range)
    print("p-order:", p_order)
    print("Ant comp:", len(d.sym_mls_funcs))
    print("root:", root)
    root_mean = root / "mean"
    mean_snapshot = Snapshot(root_mean)
    a_mean = mean_snapshot["a"]
    n_free = a_mean.shape[0]
    ns = 25 ** 2 * 5 ** 2
    print(f"ns: {ns}, n_free: {n_free}, ns <= n_free: {ns <= n_free}.")
    print("-" * 50)

    d.matrix_lsq()
    d.save_matrix_lsq()

    d.build_rb_model()
    d.save_rb_model()


def main():
    p_order = 19
    main_root = Path("DR_mls_order_analysis")
    print(datetime.now().time())
    root = main_root / f"p_order_{p_order}"
    save_mls_and_rb_model(root, p_order)
    print(datetime.now().time())


if __name__ == '__main__':
    main()
