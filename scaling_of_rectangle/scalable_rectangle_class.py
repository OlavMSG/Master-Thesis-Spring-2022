# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from itertools import product

import numpy as np
from scipy.sparse.linalg import spsolve

from assembly.get_plate import getPlateRec, getPlateTri
from scaling_of_rectangle.bilinear_quadrilateral.assembly import assemble_ints_quad
from helpers import VectorizedFunction2D, compute_a, expand_index, get_vec_from_range
from scaling_of_rectangle.linear_triangle.assembly import assemble_ints_tri
from assembly.assembly_quadrilatrial import assemble_f as assemble_f_quad
from assembly.assembly_triangle import assemble_f as assemble_f_tri
import default_constants
from pod import pod_with_energy_norm, compute_v


class ScalableRectangle:
    ref_plate = (-1, 1)
    is_jac_constant = True
    implemented_elements = ["linear triangle", "lt", "bilinear quadrilateral", "bq"]
    geo_mu_params = ["lx", "ly"]

    def __init__(self, n, f_func, get_dirichlet_edge_func=None, element="lt"):
        self.uh_rom = None
        self.last_n_rom = None
        self.v = None
        self.n_rom_max = None
        self.n_rom = None
        self.uh_full = None
        self.ints_free = None
        self.f_func_is_not_zero = None
        self.neumann_edge = None
        self.get_dirichlet_edge_func = get_dirichlet_edge_func
        self.ly = None
        self.lx = None
        self.f_load_lv_full = None
        self.ints = None
        self.rec_scale_range = default_constants.rec_scale_range
        self.n = n + 1

        self.s_mat = None
        self.ns_rom = None
        self.sigma2_vec = None
        self.z_mat = None
        self.x05 = None
        self.eps_pod = default_constants.eps_pod

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

        self.f_func_is_not_zero = True

        def default_func(x, y):
            return 0, 0

        if f_func == 0:
            self.f_func = default_func
            self.f_func_is_not_zero = False
        else:
            self.f_func = f_func

        self.p, self.tri, self.edge = self._get_plate(n, *self.ref_plate)
        self.edges()
        self.compute_free_and_expanded_edges()
        self.n_free = self.expanded_free_index.shape[0]

        print("Warning: for now: only supports constant f_funcs functions")
        print("Warning: for now: only supports constant homo. Dirichlet and Neumann BC functions")

    def set_geo_mu_params(self, lx, ly):
        self.lx = lx
        self.ly = ly

    def assemble_ints(self):
        self.ints = self._assemble_ints(self.n, self.p, self.tri)
        self._set_ints_free()

    def assemble_f(self):
        print("Warning: for now: only supports constant f_funcs functions")

        def f_func_comp_phi(x, y):
            # return self.f_func(*self.phi(x, y))
            return self.f_func(x, y)

        f_func_vec = VectorizedFunction2D(f_func_comp_phi)

        self.f_load_lv_full = self._assemble_f(self.n, self.p, self.tri, f_func_vec, self.f_func_is_not_zero)
        self._set_f_load_lv_free()

    def phi(self, x, y, lx=None, ly=None):
        if lx is None:
            lx = self.lx
        if ly is None:
            ly = self.ly
        return 0.5 * (x + lx), 0.5 * (y + ly)

    def compute_a1_and_a2_from_ints(self, lx=None, ly=None):
        # ints = (int11, int12, int21, int22, int4, int5)
        # int 3 = 0
        if lx is None:
            lx = self.lx
        if ly is None:
            ly = self.ly

        alpha0 = ly / lx
        alpha1 = lx / ly

        int1 = alpha0 * self.ints[0] + alpha1 * self.ints[1]
        int2 = alpha0 * self.ints[2] + alpha1 * self.ints[3]

        a1 = int1 + 0.5 * (int2 + self.ints[4])
        a2 = int1 + self.ints[5]

        return a1, a2

    def compute_a1(self, lx=None, ly=None):
        # ints = (int11, int12, int21, int22, int4, int5)
        # int 3 = 0
        if lx is None:
            lx = self.lx
        if ly is None:
            ly = self.ly

        alpha0 = ly / lx
        alpha1 = lx / ly

        int1 = alpha0 * self.ints[0] + alpha1 * self.ints[1]
        int2 = alpha0 * self.ints[2] + alpha1 * self.ints[3]
        a1 = int1 + 0.5 * (int2 + self.ints[4])
        return a1

    def compute_a2(self, lx=None, ly=None):
        # ints = (int11, int12, int21, int22, int4, int5)
        # int 3 = 0
        if lx is None:
            lx = self.lx
        if ly is None:
            ly = self.ly

        int1 = ly / lx * self.ints[0] + lx / ly * self.ints[1]
        a2 = int1 + self.ints[5]
        return a2

    def compute_a(self, lx, ly, e_young, nu_poisson):
        a1, a2 = self.compute_a1_and_a2_from_ints(lx, ly)
        return compute_a(e_young, nu_poisson, a1, a2)

    def set_rec_scale_range(self, rec_scale_range=None):
        if rec_scale_range is not None:
            self.rec_scale_range = rec_scale_range

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

    @property
    def geo_mu_params_range(self):
        return {"lx": self.rec_scale_range, "ly": self.rec_scale_range}

    @staticmethod
    def mls_funcs(lx, ly):
        return np.array([ly / lx, lx / ly, 1])

    def _get_dirichlet_edge(self):
        """
        Get the Dirichlet edge
        Returns
        -------
        None.
        """
        if self.get_dirichlet_edge_func is not None:
            dirichlet_edge_func_vec = np.vectorize(self.get_dirichlet_edge_func, otypes=[bool])
            index = dirichlet_edge_func_vec(self.p[self.edge, 0], self.p[self.edge, 1]).all(axis=1)
            if index.any():
                self.dirichlet_edge = self.edge[index, :]
        else:
            self.dirichlet_edge = self.edge

    def get_neumann_edge(self):
        """
        Get the Neumann edge
        Returns
        -------
        None.
        """
        if self.dirichlet_edge is None:
            self.neumann_edge = self.edge
        elif self.dirichlet_edge.shape != self.edge.shape:
            neumann_edge = np.array(list(set(map(tuple, self.edge)) - set(map(tuple, self.dirichlet_edge))))
            self.neumann_edge = neumann_edge[np.argsort(neumann_edge[:, 0]), :]

    def _are_edges_illegal(self):
        """
        Check if the edges are illegal for the solver, if so raise an error
        Raises
        ------
        EdgesAreIllegalError
            if the edges are for the solver.
        Returns
        -------
        None.
        """
        if self.get_dirichlet_edge_func is None:
            if np.all(self.neumann_edge == self.edge):
                error_text = "Only neumann conditions are not allowed, gives neumann_edge=edge, " \
                             + "please define get_dirichlet_edge_func."
                raise ValueError(error_text)
        else:
            if (self.dirichlet_edge is None) and np.all(self.neumann_edge == self.edge):
                raise ValueError("get_dirichlet_edge_func gives dirichlet_edge=None and neumann_edge=edge.")
            if (self.neumann_edge is None) and np.all(self.dirichlet_edge == self.edge):
                raise ValueError("get_dirichlet_edge_func gives dirichlet_edge=edge and neumann_edge=None.")

    def edges(self):
        """
        Get the edges
        Returns
        -------
        None.
        """
        self._get_dirichlet_edge()
        self.get_neumann_edge()
        self._are_edges_illegal()

    def _set_free_and_dirichlet_edge_index(self):
        """
        Set the free and dirichlet edge indexes
        Returns
        -------
        None.
        """
        if self.dirichlet_edge is None:
            self.dirichlet_edge_index = np.array([])
            self.free_index = np.unique(self.tri)
        else:
            # find the unique neumann _edge index
            self.dirichlet_edge_index = np.unique(self.dirichlet_edge)
            # free index is unique index minus dirichlet edge index
            self.free_index = np.setdiff1d(self.tri, self.dirichlet_edge_index)

    def _set_expanded_free_and_dirichlet_edge_index(self):
        """
        Expand the free and dirichlet edge indexes
        Returns
        -------
        None.
        """
        self.expanded_free_index = expand_index(self.free_index)
        self.expanded_dirichlet_edge_index = expand_index(self.dirichlet_edge_index)

    def compute_free_and_expanded_edges(self):
        # set self.p, self.tri, self.edge
        # self.a1_full, self.a2_full
        # self.f_load_lv_full , self.dirichlet_edge
        # optionally: self.f_load_neumann_full,  neumann_edge
        # before calling this function

        self._set_free_and_dirichlet_edge_index()
        self._set_expanded_free_and_dirichlet_edge_index()

    def _set_ints_free(self):
        self.ints_free = []
        free_xy_index = np.ix_(self.expanded_free_index, self.expanded_free_index)
        for intn in self.ints:
            self.ints_free.append(intn[free_xy_index])
        self.ints_free = tuple(self.ints_free)

    def _set_f_load_lv_free(self):
        self.f_load_lv_free = self.f_load_lv_full[self.expanded_free_index]

    def compute_a1_and_a2_from_ints_free(self, lx=None, ly=None):
        # ints = (int11, int12, int21, int22, int4, int5)
        # int 3 = 0
        if lx is None:
            lx = self.lx
        if ly is None:
            ly = self.ly

        alpha0 = ly / lx
        alpha1 = lx / ly

        int1 = alpha0 * self.ints_free[0] + alpha1 * self.ints_free[1]
        int2 = alpha0 * self.ints_free[2] + alpha1 * self.ints_free[3]

        a1 = int1 + 0.5 * (int2 + self.ints_free[4])
        a2 = int1 + self.ints_free[5]

        return a1, a2

    def compute_a_free(self, lx, ly, e_young, nu_poisson):
        a1, a2 = self.compute_a1_and_a2_from_ints_free(lx, ly)
        return compute_a(e_young, nu_poisson, a1, a2)

    def hfsolve(self, lx, ly, e_young, nu_poisson):
        self.uh_full = np.zeros(self.n * self.n * 2)
        det_j = 0.25 * lx * ly
        a_free = self.compute_a_free(lx, ly, e_young, nu_poisson).tocsr()
        f_load_lv_free = det_j * self.f_load_lv_free
        # solve
        self.uh_full[self.expanded_free_index] = spsolve(a_free, f_load_lv_free)

    def compute_a1_and_a2_from_ints_free_rom(self, lx=None, ly=None):
        # ints = (int11, int12, int21, int22, int4, int5)
        # int 3 = 0
        if lx is None:
            lx = self.lx
        if ly is None:
            ly = self.ly

        alpha0 = ly / lx
        alpha1 = lx / ly

        int1 = alpha0 * self.ints_free_rom[0] + alpha1 * self.ints_free_rom[1]
        int2 = alpha0 * self.ints_free_rom[2] + alpha1 * self.ints_free_rom[3]

        a1 = int1 + 0.5 * (int2 + self.ints_free_rom[4])
        a2 = int1 + self.ints_free_rom[5]

        return a1, a2

    def compute_a_free_rom(self, lx, ly, e_young, nu_poisson):
        a1, a2 = self.compute_a1_and_a2_from_ints_free_rom(lx, ly)
        return compute_a(e_young, nu_poisson, a1, a2)

    def rbsolve(self, lx, ly, e_young, nu_poisson, n_rom=None):
        # set n_rom to n_rom-true if it is None
        if n_rom is None:
            n_rom = self.n_rom
        # compute the rom matrices and load vectors if n_rom is different from the last used n_rom n_rom_last
        if n_rom != self.last_n_rom:
            compute_v(n_rom, self.n_free, self)
            self._compute_rom_matrices_and_vectors()
        # set last n_rom
        self.last_n_rom = self.n_rom

        self.uh_rom = np.zeros(self.n * self.n * 2)
        det_j = 0.25 * lx * ly
        a_free_rom = self.compute_a_free_rom(lx, ly, e_young, nu_poisson)
        f_load_free_rom = det_j * self.f_load_lv_free_rom
        self.uh_rom[self.expanded_free_index] = self.v @ np.linalg.solve(a_free_rom, f_load_free_rom)

    @property
    def uh_free(self):
        return self.uh_full[self.expanded_free_index]

    def _compute_rom_matrices_and_vectors(self):
        self.ints_free_rom = []
        for int_n_free in self.ints_free:
            self.ints_free_rom.append(self.v.T @ int_n_free @ self.v)
        self.ints_free_rom = tuple(self.ints_free_rom)
        self.f_load_lv_free_rom = self.v.T @ self.f_load_lv_free

    def build_rb_model(self, grid, mode):
        pod_with_energy_norm(grid, self, mode)
        compute_v(self.n_rom, self.n_free, self)
        self._compute_rom_matrices_and_vectors()

        self.last_n_rom = self.n_rom
        self.n_rom_max = np.linalg.matrix_rank(self.s_mat)
        self.uh_full = None

    def rel_error_a_rb(self, lx, ly, e_young, nu_poisson, n_rom=None, m=None, pod_mode=None):

        if n_rom is None:
            n_rom = self.n_rom
        else:
            # check if the solution matrix does not exist
            if self.s_mat is None:
                # solve new
                self.hfsolve(lx, ly, e_young, nu_poisson)
            else:
                # get solution from s_mat
                param_mat = self.param_mat(m, pod_mode)
                index = np.argwhere((param_mat[:, 0] == lx) & (param_mat[:, 1] == ly) &
                                    (param_mat[:, 2] == e_young) & (param_mat[:, 3] == nu_poisson)).ravel()
                # check if e_young and nu_poisson where not used in pod algorithm
                if len(index) == 0:
                    # solve new
                    self.hfsolve(lx, ly, e_young, nu_poisson)
                else:
                    # build from s_mat
                    self.uh_full = np.zeros(self.n * self.n * 2)
                    self.uh_full[self.expanded_free_index] = self.s_mat[:, index].flatten()
            # used n_rom
            if n_rom != self.last_n_rom:
                self.rbsolve(lx, ly, e_young, nu_poisson, n_rom=n_rom)
        # compute the error in the energy norm
        err = self.uh_full - self.uh_rom
        a_mat = self.compute_a(lx, ly, e_young, nu_poisson)
        error_a = np.sqrt(err.T @ a_mat @ err) / np.sqrt(self.uh_full.T @ a_mat @ self.uh_full)
        return error_a

    def param_mat(self, m, mode):
        rec_scale_vec = get_vec_from_range(self.rec_scale_range, m, mode)
        e_young_vec = get_vec_from_range(default_constants.e_young_range, m, mode)
        nu_poisson_vec = get_vec_from_range(default_constants.nu_poisson_range, m, mode)
        return np.array(list(product(rec_scale_vec, rec_scale_vec, e_young_vec, nu_poisson_vec)))

    @property
    def singular_values_squared_pod(self):
        return self.sigma2_vec

    @property
    def solution_matrix_rank(self):
        return np.linalg.matrix_rank(self.s_mat)
