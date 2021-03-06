# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matrix_lsq import DiskStorage, Snapshot
from fem_quadrilateral import ScalableRectangleSolver
from fem_quadrilateral.matrix_least_squares import mls_compute_from_fit, MatrixLSQ, norm
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
    max_order = 10
    print(datetime.now().time())
    d = ScalableRectangleSolver(2, 0)
    d.matrix_lsq_setup(max_order)
    print(d.sym_mls_funcs)
    print("-"*50)
    main_root = Path("SR_mls_order_analysis")
    a1_mean_norms = np.zeros(max_order)
    a2_mean_norms = np.zeros(max_order)
    f1_mean_norms = np.zeros(max_order)

    a1_mean_norms2 = np.zeros(max_order)
    a2_mean_norms2 = np.zeros(max_order)
    f1_mean_norms2 = np.zeros(max_order)

    a1_mean_rel_norms = np.zeros(max_order)
    a2_mean_rel_norms = np.zeros(max_order)
    f1_mean_rel_norms = np.zeros(max_order)
    for k, p_order in enumerate(range(1, max_order + 1)):
        root = main_root / f"p_order_{p_order}"
        print("root:", root)
        mls = MatrixLSQ(root)
        mls()
        assert len(mls.storage) != 0
        a1_norms = np.zeros(len(mls.storage))
        a2_norms = np.zeros(len(mls.storage))
        f1_norms = np.zeros(len(mls.storage))

        a1_norms2 = np.zeros(len(mls.storage))
        a2_norms2 = np.zeros(len(mls.storage))
        f1_norms2 = np.zeros(len(mls.storage))

        a1_rel_norms = np.zeros(len(mls.storage))
        a2_rel_norms = np.zeros(len(mls.storage))
        f1_rel_norms = np.zeros(len(mls.storage))
        for i, snapshot in enumerate(mls.storage):
            a1 = snapshot["a1"]
            a1_fit = mls_compute_from_fit(snapshot.data, mls.a1_list)
            a1_norms[i] = norm(a1_fit) / norm(a1)
            a1_norms2[i] = norm(a1_fit)
            a1_rel_norms[i] = norm(a1 - a1_fit) / norm(a1)
            a2 = snapshot["a2"]
            a2_fit = mls_compute_from_fit(snapshot.data, mls.a2_list)
            a2_norms[i] = norm(a2_fit) / norm(a2)
            a2_norms2[i] = norm(a2_fit)
            a2_rel_norms[i] = norm(a2 - a2_fit) / norm(a2)
            f1 = snapshot["f0"]
            f1_fit = mls_compute_from_fit(snapshot.data, mls.f0_list)
            f1_norms[i] = norm(f1_fit) / norm(f1)
            f1_norms2[i] = norm(f1_fit)
            f1_rel_norms[i] = norm(f1 - f1_fit) / norm(f1)
        a1_mean_norms[k] = np.mean(a1_norms)
        a2_mean_norms[k] = np.mean(a2_norms)
        f1_mean_norms[k] = np.mean(f1_norms)

        a1_mean_norms2[k] = np.mean(a1_norms2)
        a2_mean_norms2[k] = np.mean(a2_norms2)
        f1_mean_norms2[k] = np.mean(f1_norms2)

        a1_mean_rel_norms[k] = np.mean(a1_rel_norms)
        a2_mean_rel_norms[k] = np.mean(a2_rel_norms)
        f1_mean_rel_norms[k] = np.mean(f1_rel_norms)
    print("plotting")
    save_dict = "".join(("plots_", str(main_root)))
    Path(save_dict).mkdir(parents=True, exist_ok=True)
    x = np.arange(max_order) + 1
    plt.figure("a - 1")
    plt.title("sum a - norm")
    plt.semilogy(x, a1_mean_rel_norms, "x--", label="$\\|\\|A_1(\\mu)-\\sum_{i}g_qA_{1q}\\|\\|/\\|\\|A_1(\\mu)\\|\\|$")
    plt.semilogy(x, a2_mean_rel_norms, "x--", label="$\\|\\|A_2(\\mu)-\\sum_{i}g_qA_{2q}\\|\\|/\\|\\|A_2(\\mu)\\|\\|$")
    plt.xlabel("$p$, order")
    plt.grid()
    plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
    plt.savefig("".join((save_dict, "\\sum-a-norm.pdf")), bbox_inches='tight')
    plt.show()

    plt.figure("f - 1")
    plt.title("sum f - norm")
    plt.semilogy(x, f1_mean_rel_norms, "x--", label="$\\|\\|f_0(\\mu)-\\sum_{i}g_qf_{0q}\\|\\|/\\|\\|f_0(\\mu)\\|\\|$")
    plt.xlabel("$p$, order")
    plt.grid()
    plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
    plt.savefig("".join((save_dict, "\\sum-f-norm.pdf")), bbox_inches='tight')
    plt.show()

    plt.figure("a - 2")
    plt.title("a - relnorm")
    plt.semilogy(x, a1_mean_norms, "x--", label="$\\|\\|g_qA_{1q}\\|\\|/\\|\\|A_1(\\mu)\\|\\|$")
    plt.semilogy(x, a2_mean_norms, "x--", label="$\\|\\|g_qA_{2q}\\|\\|/\\|\\|A_2(\\mu)\\|\\|$")
    plt.xlabel("$p$, order")
    plt.grid()
    plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
    plt.savefig("".join((save_dict, "\\rel-a-norm.pdf")), bbox_inches='tight')
    plt.show()

    plt.figure("f - 2")
    plt.title("f - relnorm")
    plt.semilogy(x, f1_mean_norms, "x--", label="$\\|\\|g_qf_{0q}\\|\\|/\\|\\|f_0(\\mu)\\|\\|$")
    plt.xlabel("$p$, order")
    plt.grid()
    plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
    plt.savefig("".join((save_dict, "\\rel-f-norm.pdf")), bbox_inches='tight')
    plt.show()

    plt.figure("a - 3")
    plt.title("a - norm")
    plt.semilogy(x, a1_mean_norms2, "x--", label="$\\|\\|g_qA_{1q}\\|\\|$")
    plt.semilogy(x, a2_mean_norms2, "x--", label="$\\|\\|g_qA_{2q}\\|\\|$")
    plt.xlabel("$p$, order")
    plt.grid()
    plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
    plt.savefig("".join((save_dict, "\\a-norm.pdf")), bbox_inches='tight')
    plt.show()

    plt.figure("f - 3")
    plt.title("f - norm")
    plt.semilogy(x, f1_mean_norms2, "x--", label="$\\|\\|g_qf_{0q}\\|\\|$")
    plt.xlabel("$p$, order")
    plt.grid()
    plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
    plt.savefig("".join((save_dict, "\\f-norm.pdf")), bbox_inches='tight')
    plt.show()
    print("-" * 50)

    print(datetime.now().time())


if __name__ == '__main__':
    main()
