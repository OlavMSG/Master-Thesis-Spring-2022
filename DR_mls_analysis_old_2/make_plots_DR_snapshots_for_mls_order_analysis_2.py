# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from fem_quadrilateral import DraggableCornerRectangleSolver
from fem_quadrilateral.matrix_least_squares import MatrixLSQ, norm
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
    max_order1 = 0
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

        for k in tqdm(range(n), desc="Iterating"):
            a1_norms = np.zeros(len(mls.storage))
            a2_norms = np.zeros(len(mls.storage))
            f1_norms = np.zeros(len(mls.storage))
            for i, snapshot in enumerate(mls.storage):
                a1 = snapshot["a1"]
                a1_norms[i] = norm(snapshot.data[k] * mls.a1_list[k]) / norm(a1)
                a2 = snapshot["a2"]
                a2_norms[i] = norm(snapshot.data[k] * mls.a2_list[k]) / norm(a2)
                f1 = snapshot["f0"]
                f1_norms[i] = norm(snapshot.data[k] * mls.f0_list[k]) / norm(f1)
            a1_mean_norms[k] = np.mean(a1_norms)
            a2_mean_norms[k] = np.mean(a2_norms)
            f1_mean_norms[k] = np.mean(f1_norms)

        print("plotting")
        save_dict = "".join(("plots_", str(root)))
        Path(save_dict).mkdir(parents=True, exist_ok=True)

        x = np.arange(n)
        dimw = 0.8 / 2
        plt.figure("a - 2")
        plt.title(f"Relative contribution per term")
        plt.bar(x - dimw / 2, a1_mean_norms, dimw, label="$\\|\\|g_q(\\mu)A_{1q}\\|\\|/\\|\\|A_1(\\mu)\\|\\|$")
        plt.bar(x + dimw / 2, a2_mean_norms, dimw, label="$\\|\\|g_q(\\mu)A_{2q}\\|\\|/\\|\\|A_2(\\mu)\\|\\|$")
        plt.yscale("log")
        plt.xlabel("$q$")
        plt.grid()
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\rel-a-norm.pdf")), bbox_inches='tight')
        plt.show()

        plt.figure("f - 2")
        plt.title(f"Relative contribution per term")
        plt.bar(x, f1_mean_norms, dimw, label="$\\|\\|g_q(\\mu)f_{0q}\\|\\|/\\|\\|f_0(\\mu)\\|\\|$")
        plt.yscale("log")
        plt.xlabel("$q$")
        plt.grid()
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(loc=9, bbox_to_anchor=(0.5, -.13), ncol=2)
        plt.savefig("".join((save_dict, "\\rel-f-norm.pdf")), bbox_inches='tight')
        plt.show()
        print("-" * 50)

    print(datetime.now().time())


if __name__ == '__main__':
    main()
