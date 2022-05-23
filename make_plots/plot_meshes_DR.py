# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np
import matplotlib.pyplot as plt

from fem_quadrilateral import DraggableCornerRectangleSolver
from fem_quadrilateral.default_constants import e_young_range, nu_poisson_range


def main():
    n = 2
    d = DraggableCornerRectangleSolver(n, 0)
    mu1, mu2 = 0.2, -0.2
    d.plot_mesh(mu1, mu2)
    plt.show()

    def u_exact_func(x, y):
        return x, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    e_mean = np.mean(e_young_range)
    nu_mean = np.mean(nu_poisson_range)
    s_rec = DraggableCornerRectangleSolver(2, 0, dirichlet_bc_func=dirichlet_bc_func,
                                           bcs_are_on_reference_domain=False)
    mu1, mu2 = 0.2, -0.2
    s_rec.assemble(mu1, mu2)
    s_rec.hfsolve(e_mean, nu_mean, print_info=False)
    s_rec.hf_plot_displacement()
    # plt.savefig("plots/patch_test_DR1.pdf", bbox_inches='tight')
    plt.show()
    s_rec.plot_mesh()
    # plt.savefig("plots/patch_test_mesh_DR.pdf", bbox_inches='tight')
    plt.show()

    def u_exact_func(x, y):
        return y, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    e_mean = np.mean(e_young_range)
    nu_mean = np.mean(nu_poisson_range)
    s_rec = DraggableCornerRectangleSolver(2, 0, dirichlet_bc_func=dirichlet_bc_func,
                                           bcs_are_on_reference_domain=False)
    mu1, mu2 = 0.2, -0.2
    s_rec.assemble(mu1, mu2)
    s_rec.hfsolve(e_mean, nu_mean, print_info=False)
    s_rec.hf_plot_displacement()
    # plt.savefig("plots/patch_test_DR2.pdf", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
