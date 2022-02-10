# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import numpy as np
from itertools import product

from get_plate import getPlateRec, getPlateTri
from .bilinear_quadrilateral.assembly import assemble_ints_quad
from helpers import VectorizedFunction2D, get_vec_from_range, compute_a, get_lambda_mu
from .linear_triangle.assembly import assemble_ints_tri
from assembly_quadrilatrial import assemble_f as assemble_f_quad
from assembly_triangle import assemble_f as assemble_f_tri
from default_constants import e_young_range, nu_poisson_range, rec_scale_range


class ScalableRectangle:
    ref_plate = (-1, 1)
    is_jac_constant = True
    implemented_elements = ["linear triangle", "lt", "bilinear quadrilateral", "bq"]

    def __init__(self, n, element="linear triangle"):
        self.ly = None
        self.lx = None
        self.f_load_lv_full = None
        self.ints = None
        self.e_young_range = e_young_range
        self.nu_poisson_range = nu_poisson_range
        self.rec_scale_range = rec_scale_range

        self.n = n + 1
        if element.lower() in self.implemented_elements:
            self.element = element.lower()
        else:
            error_text = "Element " + str(element) + " is not implemented. " \
                         + "Implemented elements: " + str(self.implemented_elements)
            raise NotImplementedError(error_text)

        if self.element in ("linear triangle", "lt"):
            self._get_plate = getPlateTri
            self._assemble_ints = assemble_ints_tri
            self._assemble_f = assemble_f_tri
        else:
            self._get_plate = getPlateRec
            self._assemble_ints = assemble_ints_quad
            self._assemble_f = assemble_f_quad

        self.p, self.tri, self.edge = self._get_plate(n, *self.ref_plate)

    def set_param(self, lx, ly):
        self.lx = lx
        self.ly = ly

    def assemble_ints(self):
        self.ints = self._assemble_ints(self.n, self.p, self.tri)

    def assemble_f(self, f_func=0):
        f_func_is_not_zero = True
        if f_func == 0:
            def default_func(x, y):
                return 0, 0

            f_func = default_func
            f_func_is_not_zero = False

        def f_func_comp_phi(x, y):
            return f_func(*self.phi(x, y))

        f_func_vec = VectorizedFunction2D(f_func_comp_phi)

        self.f_load_lv_full = self._assemble_f(self.n, self.p, self.tri, f_func_vec, f_func_is_not_zero)

    def phi(self, x, y):
        return 0.5 * (x + self.lx), 0.5 * (y + self.ly)

    def compute_a1_and_a2_from_ints(self):
        # ints = (int11, int12, int21, int22, int4, int5)
        # int 3 = 0
        alpha0 = self.ly / self.lx
        alpha1 = self.lx / self.ly

        int1 = alpha0 * self.ints[0] + alpha1 * self.ints[1]
        int2 = alpha0 * self.ints[2] + alpha1 * self.ints[3]

        a1 = int1 + 0.5 * (int2 + self.ints[4])
        a2 = int1 + self.ints[5]

        return a1, a2

    def compute_a(self, lx, ly, e_young, nu_poisson):
        self.set_param(lx, ly)
        a1, a2 = self.compute_a1_and_a2_from_ints()
        return compute_a(e_young, nu_poisson, a1, a2)

    def compute_matrix_least_squares_m_and_b(self, m, rec_scale_range=None, e_young_range=None,
                                             nu_poisson_range=None, mode="uniform"):
        if rec_scale_range is not None:
            self.rec_scale_range = rec_scale_range
        if e_young_range is not None:
            self.e_young_range = e_young_range
        if nu_poisson_range is not None:
            self.nu_poisson_range = nu_poisson_range

        rec_scale_vec = get_vec_from_range(self.rec_scale_range, m, mode)
        e_young_vec = get_vec_from_range(self.e_young_range, m, mode)
        nu_poisson_vec = get_vec_from_range(self.nu_poisson_range, m, mode)

        n2d = self.n * self.n * 2
        m4 = m ** 4
        m_mat = np.zeros((m4, 6), dtype=float)
        # m_mat[:, 0] = 1 # 6->7 above
        b_mat = np.zeros((m4, n2d, n2d), dtype=float)
        for i, (lx, ly, e_young, nu_poisson) in \
                enumerate(product(rec_scale_vec, rec_scale_vec, e_young_vec, nu_poisson_vec)):
            self.set_param(lx, ly)
            mu, lam = get_lambda_mu(e_young, nu_poisson)
            m_mat[i, 0:3] = 2 * mu * self.unique_z_comp
            m_mat[i, 3:] = lam * self.unique_z_comp

            a1, a2 = self.compute_a1_and_a2_from_ints()
            b_mat[i, :, :] = 2 * mu * a1.A + lam * a2.A

        return m_mat, b_mat

    @property
    def jac(self):
        return np.array([[0.5 * self.lx, 0],
                         [0, 0.5 * self.ly]])

    @property
    def det_jac(self):
        return 0.25 * self.lx * self.ly

    @property
    def jac_inv(self):
        return np.array([[2 / self.lx, 0],
                         [0, 2 / self.ly]])

    @property
    def z(self):
        return np.array([[self.ly / self.lx, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, self.lx / self.ly]])

    @property
    def unique_z_comp(self):
        return np.array([self.ly / self.lx, self.lx / self.ly, 1])
