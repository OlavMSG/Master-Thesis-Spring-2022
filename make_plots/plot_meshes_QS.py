# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np
import matplotlib.pyplot as plt

from fem_quadrilateral import QuadrilateralSolver
from fem_quadrilateral.default_constants import e_young_range, nu_poisson_range


def main():
    n = 10
    q = QuadrilateralSolver(n, 0)
    mu1, mu2, mu3, mu4, mu5, mu6 = -0.1, -0.1, -0.1, 0.1, 0.1, 0.1
    q.plot_mesh(mu1, mu2, mu3, mu4, mu5, mu6)
    plt.show()

    def u_exact_func(x, y):
        return x, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    e_mean = np.mean(e_young_range)
    nu_mean = np.mean(nu_poisson_range)
    s_rec = QuadrilateralSolver(2, 0, dirichlet_bc_func=dirichlet_bc_func,
                                bcs_are_on_reference_domain=False)
    mu1, mu2, mu3, mu4, mu5, mu6 = -0.1, -0.1, -0.1, 0.1, 0.1, 0.1
    s_rec.assemble(mu1, mu2, mu3, mu4, mu5, mu6)
    s_rec.hfsolve(e_mean, nu_mean, print_info=False)
    s_rec.hf_plot_displacement()
    # plt.savefig("plots/patch_test_QS1.pdf", bbox_inches='tight')
    plt.show()
    s_rec.plot_mesh()
    # plt.savefig("plots/patch_test_mesh_QS.pdf", bbox_inches='tight')
    plt.show()

    def u_exact_func(x, y):
        return y, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    e_mean = np.mean(e_young_range)
    nu_mean = np.mean(nu_poisson_range)
    s_rec = QuadrilateralSolver(2, 0, dirichlet_bc_func=dirichlet_bc_func,
                                bcs_are_on_reference_domain=False)
    mu1, mu2, mu3, mu4, mu5, mu6 = -0.1, -0.1, -0.1, 0.1, 0.1, 0.1
    s_rec.assemble(mu1, mu2, mu3, mu4, mu5, mu6)
    s_rec.hfsolve(e_mean, nu_mean, print_info=False)
    s_rec.hf_plot_displacement()
    # plt.savefig("plots/patch_test_QS2.pdf", bbox_inches='tight')
    plt.show()

    rec = QuadrilateralSolver(20, 0)
    rec.plot_mesh(-0.1, -0.1, 0.1, 0.1, -0.1, -0.1)
    # plt.savefig("plots/QS20_m01.pdf", bbox_inches="tight")
    plt.show()
    rec.plot_mesh(0.1, 0.1, -0.1, -0.1, 0.1, 0.1)
    # plt.savefig("plots/QS20_p01.pdf", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
