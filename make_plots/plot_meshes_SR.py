# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np
import matplotlib.pyplot as plt

from fem_quadrilateral import ScalableRectangleSolver
from fem_quadrilateral.default_constants import e_young_range, nu_poisson_range


def main():
    n = 2
    s = ScalableRectangleSolver(n, 0)
    lx, ly = 1, 1
    s.plot_mesh(lx, ly)
    # plt.savefig("plots/ref_mesh.pdf", bbox_inches='tight')
    plt.show()

    def u_exact_func(x, y):
        return x, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    e_mean = np.mean(e_young_range)
    nu_mean = np.mean(nu_poisson_range)
    s_rec = ScalableRectangleSolver(2, 0, dirichlet_bc_func=dirichlet_bc_func,
                                    bcs_are_on_reference_domain=False)
    lx, ly = 4, 0.3
    s_rec.assemble(lx, ly)
    s_rec.hfsolve(e_mean, nu_mean, print_info=False)
    s_rec.hf_plot_displacement()
    # plt.savefig("plots/patch_test_SR1.pdf", bbox_inches='tight')
    plt.show()
    s_rec.plot_mesh()
    # plt.savefig("plots/patch_test_mesh_SR.pdf", bbox_inches='tight')
    plt.show()

    def u_exact_func(x, y):
        return y, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    e_mean = np.mean(e_young_range)
    nu_mean = np.mean(nu_poisson_range)
    s_rec = ScalableRectangleSolver(2, 0, dirichlet_bc_func=dirichlet_bc_func,
                                    bcs_are_on_reference_domain=False)
    lx, ly = 4, 0.3
    s_rec.assemble(lx, ly)
    s_rec.hfsolve(e_mean, nu_mean, print_info=False)
    s_rec.hf_plot_displacement()
    # plt.savefig("plots/patch_test_SR2.pdf", bbox_inches='tight')
    plt.show()

    rec = ScalableRectangleSolver(2, 0)
    rec.plot_mesh(0.1, 5.1)
    # plt.savefig("plots/SR2_01_51.pdf", bbox_inches="tight")
    plt.show()
    rec.plot_mesh(5.1, 0.1)
    # plt.savefig("plots/SR2_51_01.pdf", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
