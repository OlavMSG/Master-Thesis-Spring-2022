# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Optional, Tuple
from functools import partial
from scipy.sparse.linalg import spsolve
import numpy as np
from time import perf_counter

import helpers
from . import assembly
from .solution_function_class import SolutionFunctionValues2D
from .base_solver import BaseSolver
from .snapshot_saver import SnapshotSaver
from .matrix_least_squares import MatrixLSQ, mls_compute_from_fit
from .pod import PodWithEnergyNorm

symengine_is_found = (find_spec("symengine") is not None)
if symengine_is_found:
    import symengine as sym
    from symengine import Lambdify as sym_Lambdify
else:
    import sympy as sym
    from sympy import lambdify as sym_Lambdify

x1, x2 = sym.symbols("x1, x2", real=True)
sym_x_vec = sym.Matrix([x1, x2])
a1, a2, a3, a4 = sym.symbols("a1:5", real=True)
b1, b2, b3, b4 = sym.symbols("b1:5", real=True)
mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8 = sym.symbols("mu1:9", real=True)


def sym_kron_product2x2(mat1: sym.Matrix, mat2: sym.Matrix) -> sym.Matrix:
    kron = sym.Matrix(
        [[mat1[0, 0] * mat2[0, 0], mat1[0, 0] * mat2[0, 1], mat1[0, 1] * mat2[0, 0], mat1[0, 1] * mat2[0, 1]],
         [mat1[0, 0] * mat2[1, 0], mat1[0, 0] * mat2[1, 1], mat1[0, 1] * mat2[1, 0], mat1[0, 1] * mat2[1, 1]],
         [mat1[1, 0] * mat2[0, 0], mat1[1, 0] * mat2[0, 1], mat1[1, 1] * mat2[0, 0], mat1[1, 1] * mat2[0, 1]],
         [mat1[1, 0] * mat2[1, 0], mat1[1, 0] * mat2[1, 1], mat1[1, 1] * mat2[1, 0], mat1[1, 1] * mat2[1, 1]]]
    )
    return kron


class QuadrilateralSolver(BaseSolver):
    ref_plate = (0, 1)
    implemented_elements = ["linear triangle", "lt", "bilinear quadrilateral", "bq"]
    sym_phi = sym.Matrix([
        x1 + mu1 * (1 - x1) * (1 - x2) + mu3 * x1 * (1 - x2) + mu5 * x1 * x2 + mu7 * (1 - x1) * x2,
        x2 + mu2 * (1 - x1) * (1 - x2) + mu4 * x1 * (1 - x2) + mu6 * x1 * x2 + mu8 * (1 - x1) * x2
    ])
    sym_params = sym.Matrix([x1, x2, mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8])
    geo_param_range = (-0.125, 0.125)  # 1/8
    _pod: PodWithEnergyNorm
    _mls: MatrixLSQ

    @staticmethod
    def mu_to_vertices_dict():
        mu_to_vertices_dict = {"mu1": "a1", "mu2": "b1",
                               "mu3": "a2 - 1", "mu4": "b2",
                               "mu5": "a3 - 1", "mu6": "b3 - 1",
                               "mu7": "a4", "mu8": "b4 - 1"}
        print("Given the Quadrilateral with vertices in (a1, b1), (a2, b2), (a3, b3) and (a4, b4) the parameters mu1:8 "
              "are given as:")
        print(mu_to_vertices_dict)

    def __init__(self, n, f_func, dirichlet_bc_func=None, get_dirichlet_edge_func=None, neumann_bc_func=None,
                 element="bq"):

        self.uh_rom = SolutionFunctionValues2D()
        self.f2_dir_rom_list = None
        self.f1_dir_rom_list = None
        self.f0_rom_list = None
        self.a2_rom_list = None
        self.a1_rom_list = None
        self._pod_is_computed = False
        self._mls_is_computed = False
        self.saver_root = None
        self.mls_has_been_setup = False
        self.uh = SolutionFunctionValues2D()
        self.neumann_edge = np.array([])
        self.dirichlet_edge = np.array([])
        self.mls_order = None
        self.use_negative_mls_order = None
        self.n = n + 1
        self.n_full = self.n * self.n * 2
        self.is_assembled_and_free = False

        self._sym_setup()

        if element.lower() in self.implemented_elements:
            self.element = element.lower()
        else:
            error_text = "Element " + str(element) + " is not implemented. " \
                         + "Implemented elements: " + str(self.implemented_elements)
            raise NotImplementedError(error_text)

        if self.element in ("linear triangle", "lt"):
            self.nq = 4
            self.nq_y = None
        else:
            self.nq = 2
            self.nq_y = 2

        def default_func(x, y):
            return 0, 0

        # body force function
        self.f_func_non_zero = True
        if f_func == 0:
            self.f_func = helpers.VectorizedFunction2D(default_func)
            self.f_func_non_zero = False
        else:
            self.f_func = helpers.VectorizedFunction2D(
                lambda x, y:
                f_func(*self.phi(x, y, *self.geo_params)) * self.det_jac_func(x, y, *self.geo_params))

        # dirichlet bc function
        self.has_non_homo_dirichlet = False
        if dirichlet_bc_func is None:
            self.dirichlet_bc_func = helpers.VectorizedFunction2D(default_func)
        else:
            # no det_jac here implemented via a1 and a2
            self.dirichlet_bc_func = helpers.VectorizedFunction2D(
                lambda x, y: dirichlet_bc_func(*self.phi(x, y, *self.geo_params)))
            self.has_non_homo_dirichlet = True

        # get_dirichlet_edge function
        self.get_dirichlet_edge_func = get_dirichlet_edge_func
        # neumann bc function
        self.has_non_homo_neumann = False
        if self.get_dirichlet_edge_func is None:
            if neumann_bc_func is not None:
                error_text = "Have neumann and dirichlet conditions, " \
                             + "but no function giving the neumann and dirichlet edges. "
                raise ValueError(error_text)
        else:
            if neumann_bc_func is None:
                self.neumann_bc_func = helpers.VectorizedFunction2D(default_func)
            else:
                self.neumann_bc_func = helpers.VectorizedFunction2D(
                    lambda x, y:
                    neumann_bc_func(*self.phi(x, y, *self.geo_params)) * self.det_jac_func(x, y, *self.geo_params))
                self.has_non_homo_neumann = True

    def _sym_setup(self):
        self.sym_geo_params = self.sym_params[2:]
        self.phi = sym_Lambdify(self.sym_params, self.sym_phi)
        self.sym_jac = self.sym_phi.jacobian(sym_x_vec)
        self.sym_det_jac = self.sym_jac.det()
        self.det_jac_func = sym_Lambdify(self.sym_params, self.sym_det_jac)
        self.is_jac_constant = (x1 and x2 not in self.sym_jac.free_symbols)
        sym_jac_inv_det = sym.Matrix([[self.sym_jac[1, 1], - self.sym_jac[0, 1]],
                                      [-self.sym_jac[1, 0], self.sym_jac[0, 0]]])

        self.sym_z_mat = sym_kron_product2x2(sym_jac_inv_det, sym_jac_inv_det) / self.sym_det_jac

        self.z_mat_funcs = np.empty((4, 4), dtype=object)
        for i in range(4):
            for j in range(4):
                self.z_mat_funcs[i, j] = np.vectorize(sym_Lambdify(self.sym_params, self.sym_z_mat[i, j]),
                                                      otypes=[float])

    def _sym_mls_params_setup(self, ignore_jac_constant=False):
        if self.is_jac_constant and not ignore_jac_constant:
            mu_funcs = []
            for i in range(4):
                for j in range(4):
                    param = self.sym_z_mat[i, j]
                    if param not in mu_funcs and param != 0:
                        mu_funcs.append(param)
            self.sym_mls_funcs = sym.Matrix(mu_funcs).T
        else:
            # could use this on all, with negative_order = True
            # and drop is_jac_constant
            mu_funcs = self.sym_geo_params + [1]
            order_mu_params = np.ones_like(mu_funcs)
            order_mu_params[-1] = 0

            mu_params_temp = mu_funcs.copy()
            order_mu_params_temp = order_mu_params.copy()
            z_funcs = []
            z_order = []
            for _ in range(self.mls_order):
                z_funcs = []
                z_order = []
                for param1, order1 in zip(mu_params_temp, order_mu_params_temp):
                    for param2, order2 in zip(mu_funcs, order_mu_params):
                        mul = param1 * param2
                        if mul == 1:
                            mul = sym.S.One
                        if mul not in z_funcs:
                            z_funcs.append(mul)
                            z_order.append(order1 + order2)
                        if self.use_negative_mls_order:
                            div = param1 / param2
                            if div == 1:
                                div = sym.S.One
                            if div not in z_funcs:
                                z_funcs.append(div)
                                z_order.append(order1 - order2)

                mu_params_temp = z_funcs
                order_mu_params_temp = z_order
            z_funcs = np.asarray(z_funcs)
            z_order = np.asarray(z_order)
            arg_order = np.argwhere(np.abs(z_order) <= self.mls_order).ravel()
            self.sym_mls_funcs = sym.Matrix(z_funcs[arg_order].tolist()).T
        # lambdify
        self.mls_funcs = sym_Lambdify(self.sym_geo_params, self.sym_mls_funcs)

    def set_quadrature_scheme_order(self, nq: int, nq_y: Optional[int] = None):
        if self.element in ("linear triangle", "lt"):
            self.nq = nq
        else:
            self.nq = nq
            if nq_y is None:
                self.nq_y = nq
            else:
                self.nq_y = nq

    def _edges(self):
        """
        Get the edges
        Returns
        -------
        None.
        """
        # dirichlet edge
        if self.get_dirichlet_edge_func is not None:
            dirichlet_edge_func_vec = np.vectorize(self.get_dirichlet_edge_func, otypes=[bool])
            index = dirichlet_edge_func_vec(self.p[self.edge, 0], self.p[self.edge, 1]).all(axis=1)
            if index.any():
                self.dirichlet_edge = self.edge[index, :]
            else:
                error_text = "Only neumann conditions are not allowed, gives neumann_edge=edge, " \
                             + "please define get_dirichlet_edge_func."
                raise ValueError(error_text)
        else:
            self.dirichlet_edge = self.edge

        # neumann edge
        if self.dirichlet_edge.shape != self.edge.shape:
            neumann_edge = np.array(list(set(map(tuple, self.edge)) - set(map(tuple, self.dirichlet_edge))))
            self.neumann_edge = neumann_edge[np.argsort(neumann_edge[:, 0]), :]
        else:
            self.neumann_edge = np.array([])
            if (self.get_dirichlet_edge_func is not None) and np.all(self.dirichlet_edge.shape == self.edge.shape):
                raise ValueError("get_dirichlet_edge_func gives dirichlet_edge=edge and neumann_edge=None.")

    def _set_free_and_dirichlet_edge_index(self):
        """
        Set the free and dirichlet edge indexes
        Returns
        -------
        None.
        """
        # find the unique dirichlet edge index
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
        self.expanded_free_index = helpers.expand_index(self.free_index)
        self.expanded_dirichlet_edge_index = helpers.expand_index(self.dirichlet_edge_index)

    def _compute_free_and_expanded_edges(self):
        # set self.p, self.tri, self.edge
        # self.a1_full, self.a2_full
        # self.f_body_force_full , self.dirichlet_edge
        # optionally: self.f_load_neumann_full,  neumann_edge
        # before calling this function

        self._set_free_and_dirichlet_edge_index()
        self._set_expanded_free_and_dirichlet_edge_index()

    def _set_free(self):
        free_xy_index = np.ix_(self.expanded_free_index, self.expanded_free_index)
        self.a1 = self.a1_full[free_xy_index]
        self.a2 = self.a2_full[free_xy_index]
        self.f0 = self.f0_full[self.expanded_free_index]

    def _set_f_load_dirichlet(self):
        """
        Compute the lifting function

        Returns
        -------
        None.
        """
        # compute a_dirichlet
        # x and y on the dirichlet_edge
        x_vec = self.p[self.dirichlet_edge_index][:, 0]
        y_vec = self.p[self.dirichlet_edge_index][:, 1]
        # lifting function
        self.rg = helpers.FunctionValues2D.from_2xn(self.dirichlet_bc_func(x_vec, y_vec)).flatt_values

        dirichlet_xy_index = np.ix_(self.expanded_free_index, self.expanded_dirichlet_edge_index)
        self.f1_dir = self.a1_full[dirichlet_xy_index] @ self.rg
        self.f2_dir = self.a2_full[dirichlet_xy_index] @ self.rg

        # self.a1_dir = self.a1_full[dirichlet_xy_index]
        # self.a2_dir = self.a2_full[dirichlet_xy_index]

    def _assemble(self, geo_params):
        self.geo_params = geo_params
        if self.element in ("linear triangle", "lt"):
            self.p, self.tri, self.edge = assembly.triangle.get_plate.getPlate(self.n)
            self.ints, self.f0_full \
                = assembly.triangle.linear.assemble_ints_and_f_body_force(self.n, self.p, self.tri,
                                                                          self.z_mat_funcs, self.geo_params,
                                                                          self.f_func, self.f_func_non_zero,
                                                                          self.nq)
            if self.has_non_homo_neumann:
                self.f0_full += assembly.triangle.linear.assemble_f_neumann(self.n, self.p, self.neumann_edge,
                                                                            self.neumann_bc_func, self.nq)
        else:
            self.p, self.tri, self.edge = assembly.quadrilateral.get_plate.getPlate(self.n)
            self.ints, self.f0_full \
                = assembly.quadrilateral.bilinear.assemble_ints_and_f_body_force(self.n, self.p, self.tri,
                                                                                 self.z_mat_funcs, self.geo_params,
                                                                                 self.f_func, self.f_func_non_zero,
                                                                                 self.nq, self.nq_y)
            if self.has_non_homo_neumann:
                self.f0_full += assembly.quadrilateral.bilinear.assemble_f_neumann(self.n, self.p, self.neumann_edge,
                                                                                   self.neumann_bc_func, self.nq)

        self.a1_full, self.a2_full = helpers.compute_a1_and_a2(*self.ints)
        self.a1_full = self.a1_full.tocsr()
        self.a2_full = self.a2_full.tocsr()

        self._edges()
        self._compute_free_and_expanded_edges()
        self.n_free = self.expanded_free_index.shape[0]

        self._set_free()
        if self.has_non_homo_dirichlet:
            self._set_f_load_dirichlet()
        self.is_assembled_and_free = True

    def assemble(self, mu1: float, mu2: float, mu3: float, mu4: float, mu5: float, mu6: float, mu7: float, mu8: float):
        self._assemble(np.array([mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8]))

    def hfsolve(self, e_young: float, nu_poisson: float, *geo_params: Optional[float], print_info: bool = True):
        if len(geo_params) != 0:
            if not self._mls_is_computed:
                raise ValueError("Matrix LSQ is not computed, can not set geometry parameters.")
            elif len(geo_params) != len(self.sym_geo_params):
                raise ValueError(
                    f"To many geometry parameters, got {len(geo_params)} expected {len(self.sym_geo_params)}.")
            else:
                # for now, may be changed
                data = self.mls_funcs(*geo_params)
                a1_fit = mls_compute_from_fit(data, self._mls.a1_list)
                a2_fit = mls_compute_from_fit(data, self._mls.a2_list)
                f_load = mls_compute_from_fit(data, self._mls.f0_list)
                a = helpers.compute_a(e_young, nu_poisson, a1_fit, a2_fit)
                if self.has_non_homo_dirichlet:
                    f1_dir = mls_compute_from_fit(data, self._mls.f1_dir_list)
                    f2_dir = mls_compute_from_fit(data, self._mls.f2_dir_list)
                    f_load -= helpers.compute_a(e_young, nu_poisson, f1_dir, f2_dir)
        elif self.is_assembled_and_free:
            # compute a
            a = helpers.compute_a(e_young, nu_poisson, self.a1, self.a2)
            # copy the body force load vector
            f_load = self.f0.copy()
            # compute and add the dirichlet load vector if it exists
            if self.has_non_homo_dirichlet:
                f_load -= helpers.compute_a(e_young, nu_poisson, self.f1_dir, self.f2_dir)

        else:
            raise ValueError("Matrices and vectors are not assembled.")

        # initialize uh
        uh = np.zeros(self.n_full)
        start_time = perf_counter()
        # solve system
        uh[self.expanded_free_index] = spsolve(a, f_load)
        if print_info:
            print("Solved a @ uh = f_load in {:.6f} sec".format(perf_counter() - start_time))
        if self.has_non_homo_dirichlet:
            uh[self.expanded_dirichlet_edge_index] = self.rg
        # set uh, and save it in a nice way.
        self.uh = SolutionFunctionValues2D.from_1x2n(uh)
        self.uh.set_e_young_and_nu_poisson(e_young, nu_poisson)
        if print_info:
            print("Get solution by the property uh, uh_free or uh_full of the class.\n" +
                  "The property uh, extra properties values, x and y are available.")

    def rbsolve(self, e_young: float, nu_poisson: float, *geo_params: float, n_rom: Optional[int] = None,
                print_info: bool = True):
        # for now
        if not self.is_assembled_and_free:
            raise ValueError("Matrices and vectors are not assembled. Can not get p, tri, edge and rg... May change...")
        if not self._pod_is_computed:
            raise ValueError("Pod is not computed. Can not solve.")
        elif len(geo_params) != len(self.sym_geo_params):
            raise ValueError(
                f"To many geometry parameters, got {len(geo_params)} expected {len(self.sym_geo_params)}.")

        if n_rom is None:
            # for now, may be changed
            data = self.mls_funcs(*geo_params)
            a1_fit_rom = mls_compute_from_fit(data, self.a1_rom_list)
            a2_fit_rom = mls_compute_from_fit(data, self.a2_rom_list)
            f_load_rom = mls_compute_from_fit(data, self.f0_rom_list)
            a_rom = helpers.compute_a(e_young, nu_poisson, a1_fit_rom, a2_fit_rom)
            if self.has_non_homo_dirichlet:
                f1_dir_rom = mls_compute_from_fit(data, self.f1_dir_rom_list)
                f2_dir_rom = mls_compute_from_fit(data, self.f2_dir_rom_list)
                f_load_rom -= helpers.compute_a(e_young, nu_poisson, f1_dir_rom, f2_dir_rom)
        else:
            # for now, may be changed
            a1_rom_list = list(map(partial(self._pod.compute_rom, n_rom=n_rom), self._mls.a1_list))
            a2_rom_list = list(map(partial(self._pod.compute_rom, n_rom=n_rom), self._mls.a2_list))
            f0_rom_list = list(map(partial(self._pod.compute_rom, n_rom=n_rom), self._mls.f0_list))

            data = self.mls_funcs(*geo_params)
            a1_fit_rom = mls_compute_from_fit(data, a1_rom_list)
            a2_fit_rom = mls_compute_from_fit(data, a2_rom_list)
            f_load_rom = mls_compute_from_fit(data, f0_rom_list)
            a_rom = helpers.compute_a(e_young, nu_poisson, a1_fit_rom, a2_fit_rom)
            if self.has_non_homo_dirichlet:
                f1_dir_rom_list = list(map(partial(self._pod.compute_rom, n_rom=n_rom), self._mls.f1_dir_list))
                f2_dir_rom_list = list(map(partial(self._pod.compute_rom, n_rom=n_rom), self._mls.f2_dir_list))

                f1_dir_rom = mls_compute_from_fit(data, f1_dir_rom_list)
                f2_dir_rom = mls_compute_from_fit(data, f2_dir_rom_list)
                f_load_rom -= helpers.compute_a(e_young, nu_poisson, f1_dir_rom, f2_dir_rom)

        # initialize uh
        uh_rom = np.zeros(self.n_full)
        start_time = perf_counter()
        # solve and project rb solution
        if n_rom is None:
            uh_rom[self.expanded_free_index] = self._pod.v @ np.linalg.solve(a_rom, f_load_rom)
        else:
            uh_rom[self.expanded_free_index] = self._pod.get_v_mat(n_rom) @ np.linalg.solve(a_rom, f_load_rom)
        if print_info:
            print("Solved a_rom @ uh_rom = f_load_rom in {:.6f} sec".format(perf_counter() - start_time))
        if self.has_non_homo_dirichlet:  # for now
            # lifting function
            uh_rom[self.expanded_dirichlet_edge_index] = self.rg
        # set uh_rom, save it in a nice way.
        self.uh_rom = SolutionFunctionValues2D.from_1x2n(uh_rom)
        self.uh_rom.set_e_young_and_nu_poisson(e_young, nu_poisson)
        if print_info:
            print("Get solution by the property uh_rom, uh_rom_free or uh_rom_full of the class.\n" +
                  "The property uh_rom, extra properties values, x and y are available.")

    def matrix_lsq_setup(self, mls_order: int = 1, use_negative_mls_order: bool = False,
                         ignore_jac_constant: bool = False):
        self.mls_order = mls_order
        self.use_negative_mls_order = use_negative_mls_order

        self._sym_mls_params_setup(ignore_jac_constant=ignore_jac_constant)
        self.mls_has_been_setup = True

    def save_snapshots(self, root: Path, geo_grid: int,
                       geo_range: Tuple[float, float] = None,
                       mode: str = "uniform",
                       material_grid: Optional[int] = None,
                       e_young_range: Optional[Tuple[float, float]] = None,
                       nu_poisson_range: Optional[Tuple[float, float]] = None):
        self.saver_root = root
        if geo_range is not None:
            self.geo_param_range = geo_range
        saver = SnapshotSaver(self.saver_root, geo_grid, self.geo_param_range,
                              mode=mode,
                              material_grid=material_grid,
                              e_young_range=e_young_range,
                              nu_poisson_range=nu_poisson_range)
        saver(self)

    def matrix_lsq(self, root: Path):
        self._mls = MatrixLSQ(root)
        self._mls()
        self._mls_is_computed = True

    def build_rb_model(self, root: Path, eps_pod: Optional[float] = None):
        self._pod = PodWithEnergyNorm(root, eps_pod=eps_pod)
        self._pod()
        self._pod_is_computed = True
        if not self._mls_is_computed:
            # compute matrix_lsq from root
            self.matrix_lsq(root)
        # for now, may be changed
        self.a1_rom_list = list(map(self._pod.compute_rom, self._mls.a1_list))
        self.a2_rom_list = list(map(self._pod.compute_rom, self._mls.a2_list))
        self.f0_rom_list = list(map(self._pod.compute_rom, self._mls.f0_list))
        if self.has_non_homo_dirichlet:
            self.f1_dir_rom_list = list(map(self._pod.compute_rom, self._mls.f1_dir_list))
            self.f2_dir_rom_list = list(map(self._pod.compute_rom, self._mls.f2_dir_list))

    def get_u_exact(self, u_exact_func):
        return helpers.get_u_exact(self.p, lambda x, y: u_exact_func(*self.phi(x, y, *self.geo_params)))

    @property
    def n_rom(self) -> int:
        if not self._pod_is_computed:
            raise ValueError("Pod is not computed.")
        return self._pod.n_rom

    @property
    def n_rom_max(self) -> int:
        if not self._pod_is_computed:
            raise ValueError("Pod is not computed")
        return self._pod.n_rom_max

    @property
    def uh_free(self) -> np.ndarray:
        if self.uh.values is None:
            raise ValueError(
                "High fidelity Linear Elasticity 2D Problem has not been solved, can not return uh_free.")
        return self.uh.flatt_values[self.expanded_free_index]

    @property
    def uh_full(self) -> np.ndarray:
        if self.uh.values is None:
            raise ValueError(
                "High fidelity Linear Elasticity 2D Problem has not been solved, can not return uh_full.")
        return self.uh.flatt_values

    @property
    def uh_anorm2(self) -> np.ndarray:
        if self.uh.values is None:
            raise ValueError(
                "High fidelity Linear Elasticity 2D Problem has not been solved, can not return uh_anorm2.")
        return self.uh.flatt_values.T @ helpers.compute_a(self.uh.e_young, self.uh.nu_poisson, self.a1_full,
                                                          self.a2_full) @ self.uh.flatt_values

    @property
    def uh_rom_free(self) -> np.ndarray:
        if self.uh_rom.values is None:
            raise ValueError(
                "High fidelity Linear Elasticity 2D Problem has not been solved, can not return uh_free.")
        return self.uh_rom.flatt_values[self.expanded_free_index]

    @property
    def uh_rom_full(self) -> np.ndarray:
        if self.uh_rom.values is None:
            raise ValueError(
                "High fidelity Linear Elasticity 2D Problem has not been solved, can not return uh_full.")
        return self.uh_rom.flatt_values


class DraggableCornerRectangleSolver(QuadrilateralSolver):
    sym_phi = sym.expand(QuadrilateralSolver.sym_phi.subs({mu1: 0, mu2: 0,
                                                           mu3: 0, mu4: 0,
                                                           mu5: mu1, mu6: mu2,
                                                           mu7: 0, mu8: 0}))
    sym_params = sym.Matrix([x1, x2, mu1, mu2])
    geo_param_range = (-0.25, 0.25)  # 1/4

    @staticmethod
    def mu_to_vertices_dict():
        mu_to_vertices_dict = {"mu1": "a3 - 1", "mu2": "b3 - 1"}
        print("Given the \"Rectangle\" with vertices in (0, 0), (0, 1), (a3, b3) and (0, 1) the parameters mu1:2 "
              "are given as:")
        print(mu_to_vertices_dict)

    def __init__(self, n, f_func, dirichlet_bc_func=None, get_dirichlet_edge_func=None, neumann_bc_func=None,
                 element="bq"):
        super().__init__(n, f_func, dirichlet_bc_func, get_dirichlet_edge_func, neumann_bc_func, element)

    def assemble(self, mu1: float, mu2: float, **kwargs: float):
        self._assemble(np.array([mu1, mu2]))


class ScalableRectangleSolver(QuadrilateralSolver):
    sym_phi = sym.expand(QuadrilateralSolver.sym_phi.subs({mu1: 0, mu2: 0,
                                                           mu3: mu1 - 1, mu4: 0,
                                                           mu5: mu1 - 1, mu6: mu2 - 1,
                                                           mu7: 0, mu8: mu2 - 1}))
    sym_params = sym.Matrix([x1, x2, mu1, mu2])
    geo_param_range = (0.1, 5)

    @staticmethod
    def mu_to_vertices_dict():
        mu_to_vertices_dict = {"mu1": "Lx", "mu2": "Ly"}
        print("Given the \"Rectangle\" with vertices in (0, 0), (0, Lx), (Lx, Ly) and (0, Ly) the parameters mu1:2 "
              "are given as:")
        print(mu_to_vertices_dict)

    def __init__(self, n, f_func, dirichlet_bc_func=None, get_dirichlet_edge_func=None, neumann_bc_func=None,
                 element="bq"):
        super().__init__(n, f_func, dirichlet_bc_func, get_dirichlet_edge_func, neumann_bc_func, element)

    def assemble(self, mu1: float, mu2: float, **kwargs: float):
        self._assemble(np.array([mu1, mu2]))
