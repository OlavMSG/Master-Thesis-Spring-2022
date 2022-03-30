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
    for p_order in range(1, max_order + 1):
        main_root = Path("SR_mls_order_analysis")
        root = main_root / f"p_order_{p_order}"
        print("root:", root)
        mls = MatrixLSQ(root)
        mls()
        assert len(mls.storage) != 0
        n = mls.storage[1].data.shape[0]
        a1_mean_norms = np.zeros(n)
        a2_mean_norms = np.zeros(n)
        f1_mean_norms = np.zeros(n)
        f2_mean_norms = np.zeros(n)

        a1_mean_norms2 = np.zeros(n)
        a2_mean_norms2 = np.zeros(n)
        f1_mean_norms2 = np.zeros(n)
        f2_mean_norms2 = np.zeros(n)

        a1_mean_rel_norms = np.zeros(n)
        a2_mean_rel_norms = np.zeros(n)
        f1_mean_rel_norms = np.zeros(n)
        f2_mean_rel_norms = np.zeros(n)
        for k in tqdm(range(n), desc="Iterating"):
            a1_norms = np.zeros(len(mls.storage))
            a2_norms = np.zeros(len(mls.storage))
            f1_norms = np.zeros(len(mls.storage))
            f2_norms = np.zeros(len(mls.storage))

            a1_norms2 = np.zeros(len(mls.storage))
            a2_norms2 = np.zeros(len(mls.storage))
            f1_norms2 = np.zeros(len(mls.storage))
            f2_norms2 = np.zeros(len(mls.storage))

            a1_rel_norms = np.zeros(len(mls.storage))
            a2_rel_norms = np.zeros(len(mls.storage))
            f1_rel_norms = np.zeros(len(mls.storage))
            f2_rel_norms = np.zeros(len(mls.storage))
            for i, snapshot in enumerate(mls.storage):
                a1 = snapshot["a1"]
                a1_norms[i] = norm(snapshot.data[k] * mls.a1_list[k]) / norm(a1)
                a1_norms2[i] = norm(snapshot.data[k] * mls.a1_list[k])
                a1_rel_norms[i] = norm(a1 - snapshot.data[k] * mls.a1_list[k]) / norm(a1)
                a2 = snapshot["a2"]
                a2_norms[i] = norm(snapshot.data[k] * mls.a2_list[k]) / norm(a2)
                a2_norms2[i] = norm(snapshot.data[k] * mls.a2_list[k])
                a2_rel_norms[i] = norm(a2 - snapshot.data[k] * mls.a2_list[k]) / norm(a2)
                f1 = snapshot["f1_dir"]
                f1_norms[i] = norm(snapshot.data[k] * mls.f1_dir_list[k]) / norm(f1)
                f1_norms2[i] = norm(snapshot.data[k] * mls.f1_dir_list[k])
                f1_rel_norms[i] = norm(f1 - snapshot.data[k] * mls.f1_dir_list[k]) / norm(f1)
                f2 = snapshot["f2_dir"]
                f2_norms[i] = norm(snapshot.data[k] * mls.f2_dir_list[k]) / norm(f2)
                f2_norms2[i] = norm(snapshot.data[k] * mls.f2_dir_list[k])
                f2_rel_norms[i] = norm(f2 - snapshot.data[k] * mls.f2_dir_list[k]) / norm(f2)
            a1_mean_norms[k] = np.mean(a1_norms)
            a2_mean_norms[k] = np.mean(a2_norms)
            f1_mean_norms[k] = np.mean(f1_norms)
            f2_mean_norms[k] = np.mean(f2_norms)

            a1_mean_norms2[k] = np.mean(a1_norms2)
            a2_mean_norms2[k] = np.mean(a2_norms2)
            f1_mean_norms2[k] = np.mean(f1_norms2)
            f2_mean_norms2[k] = np.mean(f2_norms2)

            a1_mean_rel_norms[k] = np.mean(a1_rel_norms)
            a2_mean_rel_norms[k] = np.mean(a2_rel_norms)
            f1_mean_rel_norms[k] = np.mean(f1_rel_norms)
            f2_mean_rel_norms[k] = np.mean(f2_rel_norms)
        print("plotting")
        save_dict = "".join(("plots_", str(root)))
        Path(save_dict).mkdir(parents=True, exist_ok=True)

        plt.figure("a - 1")
        plt.title(f"a - sum - norm - order = {p_order}")
        plt.semilogy(a1_mean_rel_norms, "x--", label="$\\|\\|A_1(\\mu)-g_qA_{1q}\\|\\|/\\|\\|A_1(\\mu)\\|\\|$")
        plt.semilogy(a2_mean_rel_norms, "x--", label="$\\|\\|A_2(\\mu)-g_qA_{2q}\\|\\|/\\|\\|A_2(\\mu)\\|\\|$")
        plt.xlabel("$q$")
        plt.grid()
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\a-sum-norm.pdf")), bbox_inches='tight')
        plt.show()

        plt.figure("f - 1")
        plt.title(f"f - sum - norm - order = {p_order}")
        plt.semilogy(f1_mean_rel_norms, "x--", label="$\\|\\|f_1(\\mu)-g_qf_{1q}\\|\\|/\\|\\|f_1(\\mu)\\|\\|$")
        plt.semilogy(f2_mean_rel_norms, "x--", label="$\\|\\|f_2(\\mu)-g_qf_{2q}\\|\\|/\\|\\|f_2(\\mu)\\|\\|$")
        plt.xlabel("$q$")
        plt.grid()
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\f-sum-norm.pdf")), bbox_inches='tight')
        plt.show()

        plt.figure("a - 2")
        plt.title(f"a - relnorm - order = {p_order}")
        plt.semilogy(a1_mean_norms, "x--", label="$\\|\\|g_qA_{1q}\\|\\|/\\|\\|A_1(\\mu)\\|\\|$")
        plt.semilogy(a2_mean_norms, "x--", label="$\\|\\|g_qA_{2q}\\|\\|/\\|\\|A_2(\\mu)\\|\\|$")
        plt.xlabel("$q$")
        plt.grid()
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\rel-a-norm.pdf")), bbox_inches='tight')
        plt.show()

        plt.figure("f - 2")
        plt.title(f"f - relnorm - order = {p_order}")
        plt.semilogy(f1_mean_norms, "x--", label="$\\|\\|g_qf_{1q}\\|\\|/\\|\\|f_1(\\mu)\\|\\|$")
        plt.semilogy(f2_mean_norms, "x--", label="$\\|\\|g_qf_{2q}\\|\\|/\\|\\|f_2(\\mu)\\|\\|$")
        plt.xlabel("$q$")
        plt.grid()
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\rel-f-norm.pdf")), bbox_inches='tight')
        plt.show()

        plt.figure("a - 3")
        plt.title(f"a - norm - order = {p_order}")
        plt.semilogy(a1_mean_norms2, "x--", label="$\\|\\|g_qA_{1q}\\|\\|$")
        plt.semilogy(a2_mean_norms2, "x--", label="$\\|\\|g_qA_{2q}\\|\\|$")
        plt.xlabel("$q$")
        plt.grid()
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\a-norm.pdf")), bbox_inches='tight')
        plt.show()

        plt.figure("f - 3")
        plt.title(f"f - norm - order = {p_order}")
        plt.semilogy(f1_mean_norms2, "x--", label="$\\|\\|g_qf_{1q}\\|\\|$")
        plt.semilogy(f2_mean_norms2, "x--", label="$\\|\\|g_qf_{2q}\\|\\|$")
        plt.xlabel("$q$")
        plt.grid()
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\f-norm.pdf")), bbox_inches='tight')
        plt.show()
        print("-" * 50)

    print(datetime.now().time())


if __name__ == '__main__':
    main()

