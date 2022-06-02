# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from fem_quadrilateral import DraggableCornerRectangleSolver, default_constants
from fem_quadrilateral.default_constants import e_young_range, nu_poisson_range

# rho_steal = 8e3  # kg/m^3
alpha = 8e3 * 100 * 9.81 * 0.01  # N/m^2


# Example 2: Gravity in 2D
def f(x, y):
    return alpha, 0


def clamped_bc(x, y):
    return abs(x) <= default_constants.default_tol


def main():
    p_order = 19
    main_root = Path("DR_mls_order_analysis")
    root = main_root / f"p_order_{p_order}"
    print(root)

    e_mean = np.mean(e_young_range)
    nu_mean = np.mean(nu_poisson_range)
    n = 90

    d = DraggableCornerRectangleSolver(n, f_func=f, get_dirichlet_edge_func=clamped_bc,
                                       bcs_are_on_reference_domain=False)
    mu1, mu2 = 0.2, -0.2
    d.assemble(mu1, mu2)
    d.hfsolve(e_mean, nu_mean)
    d.hf_plot_displacement()
    # plt.savefig("plots_pod/DR_final_hf_dis_pm02.pdf", bbox_inches="tight")
    plt.show()
    d.hf_von_mises_stress()
    levels = np.linspace(0, 200_000, 25)
    d.hf_plot_von_mises(levels=levels)
    # plt.savefig("plots_pod/DR_final_hf_von_miss_pm02.pdf", bbox_inches="tight")
    plt.show()
    # get RB data
    d = DraggableCornerRectangleSolver.from_root(root)
    d.matrix_lsq_setup()
    d.rbsolve(e_mean, nu_mean, mu1, mu2)
    d.rb_plot_displacement()
    # plt.savefig("plots_pod/DR_final_rb_dis_pm02.pdf", bbox_inches="tight")
    plt.show()
    d.rb_von_mises_stress()
    d.rb_plot_von_mises(levels=levels)
    # plt.savefig("plots_pod/DR_final_rb_von_miss_pm02.pdf", bbox_inches="tight")
    plt.show()

    for i in [1, 2, 3, 4, 5, 6, 7, 8]:
        d.rb_plot_pod_mode(i)
        # plt.savefig(f"plots_pod/DR_final_rb_pod_mode_{i}.pdf", bbox_inches="tight")
        plt.xlim(0.94, 1.01)
        # plt.savefig(f"plots_pod/DR_final_rb_pod_mode_{i}_zoom.pdf", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    main()
