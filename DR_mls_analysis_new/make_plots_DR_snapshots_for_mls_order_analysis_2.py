# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matrix_lsq import DiskStorage, Snapshot
from fem_quadrilateral import DraggableCornerRectangleSolver
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
    tol = 5e-3
    max_order1 = 10
    max_order = 25
    print(datetime.now().time())
    d = DraggableCornerRectangleSolver(2, 0)
    d.set_geo_param_range((-0.49, 0.49))
    d.matrix_lsq_setup(max_order)
    print(d.sym_mls_funcs)
    print("-" * 50)
    for p_order in range(max_order1 + 1, max_order + 1):
        main_root = Path("DR_mls_order_analysis")
        root = main_root / f"p_order_{p_order}"
        print("root:", root)
        mls = MatrixLSQ(root)
        mls()
        assert len(mls.storage) != 0
        n = mls.storage[1].data.shape[0]
        a1_mean_norms = np.zeros(n)
        a2_mean_norms = np.zeros(n)
        f1_mean_norms = np.zeros(n)

        a1_mean_norms2 = np.zeros(n)
        a2_mean_norms2 = np.zeros(n)
        f1_mean_norms2 = np.zeros(n)

        a1_mean_rel_norms = np.zeros(n)
        a2_mean_rel_norms = np.zeros(n)
        f1_mean_rel_norms = np.zeros(n)

        a1_max_rel_norms = np.zeros(n)
        a2_max_rel_norms = np.zeros(n)
        f1_max_rel_norms = np.zeros(n)

        a1_min_rel_norms = np.zeros(n)
        a2_min_rel_norms = np.zeros(n)
        f1_min_rel_norms = np.zeros(n)
        for k in tqdm(range(n), desc="Iterating"):
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
                a1_norms[i] = norm(snapshot.data[k] * mls.a1_list[k]) / norm(a1)
                a1_norms2[i] = norm(mls.a1_list[k])
                a1_rel_norms[i] = norm(a1 - snapshot.data[k] * mls.a1_list[k]) / norm(a1)
                a2 = snapshot["a2"]
                a2_norms[i] = norm(snapshot.data[k] * mls.a2_list[k]) / norm(a2)
                a2_norms2[i] = norm(mls.a2_list[k])
                a2_rel_norms[i] = norm(a2 - snapshot.data[k] * mls.a2_list[k]) / norm(a2)
                f1 = snapshot["f0"]
                f1_norms[i] = norm(snapshot.data[k] * mls.f0_list[k]) / norm(f1)
                f1_norms2[i] = norm(mls.f0_list[k])
                f1_rel_norms[i] = norm(f1 - snapshot.data[k] * mls.f0_list[k]) / norm(f1)
            a1_mean_norms[k] = np.mean(a1_norms)
            a2_mean_norms[k] = np.mean(a2_norms)
            f1_mean_norms[k] = np.mean(f1_norms)

            a1_mean_norms2[k] = np.mean(a1_norms2)
            a2_mean_norms2[k] = np.mean(a2_norms2)
            f1_mean_norms2[k] = np.mean(f1_norms2)

            a1_mean_rel_norms[k] = np.mean(a1_rel_norms)
            a2_mean_rel_norms[k] = np.mean(a2_rel_norms)
            f1_mean_rel_norms[k] = np.mean(f1_rel_norms)

            a1_mean_rel_norms[k] = np.mean(a1_rel_norms)
            a2_mean_rel_norms[k] = np.mean(a2_rel_norms)
            f1_mean_rel_norms[k] = np.mean(f1_rel_norms)

            a1_max_rel_norms[k] = np.max(a1_rel_norms)
            a2_max_rel_norms[k] = np.max(a2_rel_norms)
            f1_max_rel_norms[k] = np.max(f1_rel_norms)

            a1_min_rel_norms[k] = np.min(a1_rel_norms)
            a2_min_rel_norms[k] = np.min(a2_rel_norms)
            f1_min_rel_norms[k] = np.min(f1_rel_norms)


        print("plotting")
        save_dict = "".join(("plots_", str(root)))
        Path(save_dict).mkdir(parents=True, exist_ok=True)
        x = np.arange(n)
        plt.figure("a1 - 1")
        plt.title(f"a1 - sum - norm - order = {p_order}")
        plt.fill_between(x, a1_min_rel_norms, a1_max_rel_norms, color="b", alpha=0.5)
        plt.semilogy(a1_mean_rel_norms, "x--", color="r",
                     label="$\\|\\|A_1(\\mu)-g_qA_{1q}\\|\\|/\\|\\|A_1(\\mu)\\|\\|$")
        plt.xlabel("$q$")
        plt.grid()
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\a1-sum-norm.pdf")), bbox_inches='tight')
        plt.show()

        plt.figure("a2 - 1")
        plt.title(f"a2 - sum - norm - order = {p_order}")
        plt.fill_between(x, a2_min_rel_norms, a2_max_rel_norms, color="b", alpha=0.5)
        plt.semilogy(a2_mean_rel_norms, "x--", color="r",
                     label="$\\|\\|A_2(\\mu)-g_qA_{2q}\\|\\|/\\|\\|A_2(\\mu)\\|\\|$")
        plt.xlabel("$q$")
        plt.grid()
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\a2-sum-norm.pdf")), bbox_inches='tight')
        plt.show()

        plt.figure("f1 - 1")
        plt.title(f"f0 - sum - norm - order = {p_order}")
        plt.fill_between(x, f1_min_rel_norms, f1_max_rel_norms, color="b", alpha=0.5)
        plt.semilogy(f1_mean_rel_norms, "x--", color="r",
                     label="$\\|\\|f_0(\\mu)-g_qf_{0q}\\|\\|/\\|\\|f_0(\\mu)\\|\\|$")
        plt.xlabel("$q$")
        plt.grid()
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\f0-sum-norm.pdf")), bbox_inches='tight')
        plt.show()


        plt.figure("a1z - 1")
        plt.title(f"a1z - sum - norm - order = {p_order}")
        plt.fill_between(x, a1_min_rel_norms, a1_max_rel_norms, color="b", alpha=0.5)
        plt.semilogy(a1_mean_rel_norms, "x--", color="r",
                     label="$\\|\\|A_1(\\mu)-g_qA_{1q}\\|\\|/\\|\\|A_1(\\mu)\\|\\|$")
        plt.xlabel("$q$")
        plt.grid()
        plt.ylim(1-tol, 1+tol)
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\a1z-sum-norm.pdf")), bbox_inches='tight')
        plt.show()

        plt.figure("a2z - 1")
        plt.title(f"a2z - sum - norm - order = {p_order}")
        plt.fill_between(x, a2_min_rel_norms, a2_max_rel_norms, color="b", alpha=0.5)
        plt.semilogy(a2_mean_rel_norms, "x--", color="r",
                     label="$\\|\\|A_2(\\mu)-g_qA_{2q}\\|\\|/\\|\\|A_2(\\mu)\\|\\|$")
        plt.xlabel("$q$")
        plt.grid()
        plt.ylim(1-tol, 1+tol)
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\a2z-sum-norm.pdf")), bbox_inches='tight')
        plt.show()

        plt.figure("f1z - 1")
        plt.title(f"f0z - sum - norm - order = {p_order}")
        plt.fill_between(x, f1_min_rel_norms, f1_max_rel_norms, color="b", alpha=0.5)
        plt.semilogy(f1_mean_rel_norms, "x--", color="r",
                     label="$\\|\\|f_0(\\mu)-g_qf_{0q}\\|\\|/\\|\\|f_0(\\mu)\\|\\|$")
        plt.xlabel("$q$")
        plt.grid()
        plt.ylim(1-tol, 1+tol)
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\f0z-sum-norm.pdf")), bbox_inches='tight')
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
        plt.semilogy(f1_mean_norms, "x--", label="$\\|\\|g_qf_{0q}\\|\\|/\\|\\|f_0(\\mu)\\|\\|$")
        plt.xlabel("$q$")
        plt.grid()
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\rel-f-norm.pdf")), bbox_inches='tight')
        plt.show()

        plt.figure("a - 3")
        plt.title(f"a - norm - order = {p_order}")
        plt.semilogy(a1_mean_norms2, "x--", label="$\\|\\|A_{1q}\\|\\|$")
        plt.semilogy(a2_mean_norms2, "x--", label="$\\|\\|A_{2q}\\|\\|$")
        plt.xlabel("$q$")
        plt.grid()
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\a-norm.pdf")), bbox_inches='tight')
        plt.show()

        plt.figure("f - 3")
        plt.title(f"f - norm - order = {p_order}")
        plt.semilogy(f1_mean_norms2, "x--", label="$\\|\\|f_{0q}\\|\\|$")
        plt.xlabel("$q$")
        plt.grid()
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\f-norm.pdf")), bbox_inches='tight')
        plt.show()
        print("-" * 50)

    print(datetime.now().time())


if __name__ == '__main__':
    main()
