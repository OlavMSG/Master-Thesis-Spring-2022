# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations

from importlib.util import find_spec
from itertools import product, repeat
from pathlib import Path
from typing import Optional, Tuple, Callable, Union, Iterable

from scipy.sparse.linalg import spsolve
import numpy as np
from time import perf_counter
from scipy.special import legendre
import tqdm

import helpers
from . import assembly
from .solution_function_class import SolutionFunctionValues2D
from .base_solver import BaseSolver
from .snapshot_saver import SnapshotSaver
from .matrix_least_squares import MatrixLSQ, mls_compute_from_fit
from .pod import PodWithEnergyNorm
from .error_computers import HfErrorComputer, RbErrorComputer

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
mu1, mu2, mu3, mu4, mu5, mu6 = sym.symbols("mu1:7", real=True)
lx, ly = sym.symbols("Lx, Ly", real=True)
x0, y0 = sym.symbols("x0, y0", real=True)


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
    """sym_phi = sym.Matrix([
        x0 + x1 + mu1 * (1 - x1) * (1 - x2) + mu3 * x1 * (1 - x2) + mu5 * x1 * x2 + mu7 * (1 - x1) * x2,
        y0 + x2 + mu2 * (1 - x1) * (1 - x2) + mu4 * x1 * (1 - x2) + mu6 * x1 * x2 + mu8 * (1 - x1) * x2
    ])"""
    sym_phi = sym.Matrix([
        x0 + x1 + mu1 * x1 * (1 - x2) + mu3 * x1 * x2 + mu5 * (1 - x1) * x2,
        y0 + x2 + mu2 * x1 * (1 - x2) + mu4 * x1 * x2 + mu6 * (1 - x1) * x2
    ])
    sym_params = sym.Matrix([x1, x2, mu1, mu2, mu3, mu4, mu5, mu6])
    geo_param_range = (-0.16, 0.16)  # < 1/6 = 0.1666...
    _pod: PodWithEnergyNorm
    _mls: MatrixLSQ

    @staticmethod
    def mu_to_vertices_dict():
        mu_to_vertices_dict = {"mu1": "a2 - 1", "mu2": "b2",
                               "mu3": "a3 - 1", "mu4": "b3 - 1",
                               "mu5": "a4", "mu6": "b4 - 1"}
        print("(x0, y0): are the coordinates of the lower left corner, default (0,0)")
        print("Given the Quadrilateral centered in (0, 0) with vertices in (0, 0), (a2, b2), (a3, b3) and (a4, "
              "b4) the parameters mu1:8 are given as:")
        print(mu_to_vertices_dict)

    def __init__(self, n: int, f_func: Union[Callable, int],
                 dirichlet_bc_func: Optional[Callable] = None, get_dirichlet_edge_func: Optional[Callable] = None,
                 neumann_bc_func: Optional[Callable] = None,
                 element: str = "bq", x0: float = 0, y0: float = 0):

        self._use_experimental_expansion_in_mls = False
        self._a1_set_n_rom_list = None
        self._a2_set_n_rom_list = None
        self._f0_set_n_rom_list = None
        self._f1_dir_set_n_rom_list = None
        self._f2_dir_set_n_rom_list = None
        self._last_n_rom = -1
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
        self.mls_order = 1
        self.use_negative_mls_order = None
        self._n = n + 1
        self.n_full = self._n * self._n * 2
        self.is_assembled_and_free = False
        self.is_assembled_and_free_from_root = False

        self.lower_left_corner = (x0, y0)
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

    def _from_root(self):
        from matrix_lsq import Snapshot
        root_mean = self.saver_root / "mean"
        mean_snapshot = Snapshot(root_mean)
        self.geo_param_range = mean_snapshot["ranges"][0]
        self.p, self.tri, self.edge = mean_snapshot["p"], mean_snapshot["tri"], mean_snapshot["edge"]
        self.dirichlet_edge = mean_snapshot["dirichlet_edge"]
        self.neumann_edge = mean_snapshot["neumann_edge"]
        self.has_non_homo_dirichlet = (self.saver_root / "object-0" / "obj-f1_dir.npy").exists()
        if self.has_non_homo_dirichlet:
            self.rg = mean_snapshot["rg"]
        # note here it does not matter if we have non-homogeneous or not, or if we have neumann conditions
        mls_and_llc = mean_snapshot["mls_order_and_llc"]
        self.mls_order = int(mls_and_llc[0])
        # get lower_left_corner
        self.lower_left_corner = tuple(mls_and_llc[1:])
        # update phi, note do not have f_func, dirichlet_bc and neumann_bc func so 0,0 was used in cls() call.
        self.sym_phi = self.sym_phi + sym.Matrix(self.lower_left_corner)
        self.phi = sym_Lambdify(self.sym_params, self.sym_phi)
        # compute free indexes.
        self._compute_free_and_expanded_edges()
        self.n_full = self.p.shape[0] * 2
        self._n = int(np.sqrt(self.p.shape[0]))
        self.is_assembled_and_free_from_root = True

    @classmethod
    def from_root(cls, root: Path):
        out = cls(2, 0)
        out.saver_root = root
        out._from_root()
        return out

    @property
    def n(self) -> int:
        return self._n - 1

    def _sym_setup(self):
        self.sym_phi = self.sym_phi.subs({x0: self.lower_left_corner[0], y0: self.lower_left_corner[1]})
        self.sym_geo_params = self.sym_params[2:]
        self.phi = sym_Lambdify(self.sym_params, self.sym_phi)
        self.sym_jac = self.sym_phi.jacobian(sym_x_vec)
        self.sym_det_jac = self.sym_jac.det().expand()
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

    def _sym_mls_params_setup(self):
        ant = len(self.sym_geo_params) + 1

        # looking at Z =  top_z / det(jac)
        # 1 / det(jac) has gives a rational of form 1/(k + c*x)
        # Which has the Taylor expansion
        # 1 / (k + c*x) = 1/k - cx/k^2 + c^2x^2/k^3 - ... + ...
        # for |x|<|k/c|

        # c here takes the form a_0*1 + sum (b_i * mu_i)
        # so c^order takes the form a_0*1 + sum (b_i * mu_i) + sum (c_ij * mu_i * mu_j)
        # + sum (d_ijk * mu_i * mu_j * mu_k) + ...

        # Furthermore, the numerator in Z takes the form
        # a_0*1 + sum (b_i * mu_i) + sum (c_ij * mu_i * mu_j), i.e order 2

        # looking at f(phi) * det(jac)
        # f(phi) to the order m takes the same form as c^order above
        # det(jac) takes the form
        # a_0*1 + sum (b_i * mu_i) + sum_{i!=j} (c_ij * mu_i * mu_j)
        # and is contained in order 2

        # putting it all together we have
        # k, c and top_z for Z matrix, and
        # c and (det_jac) top_z for f(phi) * det(jac).
        # conclusion: we do not nees top_z and get_jac, if we compute c 2 orders higher

        # get k_term from det(jac) by setting x1=x2=0
        k_term = self.sym_det_jac.subs({x1: 0, x2: 0})
        k_orders = [np.zeros(ant, dtype=int)]
        if k_term != 1:
            if self.is_jac_constant:
                # Jacobian is constant, i.e k is constant
                if len(k_term.args) == len(self.sym_geo_params) and \
                        all(k_arg in self.sym_geo_params for k_arg in self.sym_geo_params):
                    # k has one component, that is PI_{i=1} mu_i
                    k_order = - np.ones(ant, dtype=int)
                    k_order[-1] = 0  # set total to zero, we only have one term
                    k_orders.append(k_order)
                else:
                    # k has multiple components, form a_0*1 + sum (b_i * mu_i) + sum_{i!=j} (c_ij * mu_i * mu_j)
                    k_order = np.zeros(ant, dtype=int)
                    k_order[-1] = -2  # set total to neg two, we only have one term
                    k_orders.append(k_order)

            elif len(k_term.args) == len(self.sym_geo_params) and \
                    all(k_arg in self.sym_geo_params for k_arg in self.sym_geo_params):
                # Jacobian is not constant and k has one component, that is PI_{i=1} mu_i
                for order in range(self.mls_order):
                    k_order = -(order + 1) * np.ones(ant, dtype=int)
                    k_order[-1] = 0  # set total to zero, we only have one term
                    k_orders.append(k_order)
            else:
                # not tested
                # Jacobian is not constant and k has multiple components,
                # form a_0*1 + sum (b_i * mu_i) + sum_{i!=j} (c_ij * mu_i * mu_j)
                for order in range(self.mls_order // 2):
                    k_order = np.zeros(ant, dtype=int)
                    k_order[-1] = - 2 * (order + 1)  # set total, we have multiple terms
                    k_orders.append(k_order)
        # experimental expansions
        if self._use_experimental_expansion_in_mls:
            x1_term = self.sym_det_jac.subs({x1: 1, x2: 0}) - k_term
            x2_term = self.sym_det_jac.subs({x1: 0, x2: 1}) - k_term
            if (k_term == 1) and (x1_term in self.sym_geo_params) and (x2_term in self.sym_geo_params):
                # add 1 / (x1_term * x2_term)
                # here know len(self.sym_geo_params)=2
                k_order = - np.ones(ant, dtype=int)
                k_order[-1] = 0  # set total to zero, we only have one term
                k_orders.append(k_order)

        k_orders = np.unique(k_orders, axis=0)

        # get c
        c = np.array(list(product(*repeat(np.arange(self.mls_order + 2), ant - 1))))
        c = c[np.argwhere(np.sum(c, axis=1) <= self.mls_order + 2).ravel()]
        c_orders = np.zeros((c.shape[0], c.shape[1] + 1), dtype=int)
        c_orders[:, :-1] = c

        # put it all together
        mls_orders = [np.zeros(ant, dtype=int)]
        # print(c_orders)
        # print(k_orders)
        for order1 in k_orders:
            for order2 in c_orders:
                sum_order = order1 + order2
                mls_orders.append(sum_order)
        mls_orders = np.unique(mls_orders, axis=0)

        # get orders to use
        arg_order = np.array(list(np.all(np.abs(order) <= self.mls_order)
                                  and ((np.where(order > 0, order, 0).sum() <= self.mls_order)
                                       and (-np.where(order < 0, order, 0).sum()) <= self.mls_order)
                                  for order in mls_orders))

        mls_orders = mls_orders[arg_order]
        # print(mls_orders)
        p_list = [legendre(p).coeffs[::-1] for p in range(self.mls_order + 1)]
        # make the mls_funcs
        sym_mls_funcs = sym.Matrix([sym.S.One] * len(mls_orders))
        # shift input to polynomials to (a, b)
        a_plus_b = self.geo_param_range[0] + self.geo_param_range[1]
        b_minus_a_1 = 1 / (self.geo_param_range[1] - self.geo_param_range[0])

        def shift(x):
            return (2 * x - a_plus_b) * b_minus_a_1

        for n, order in tqdm.tqdm(enumerate(mls_orders), desc="computing mls functions"):
            for i, mu_i in enumerate(self.sym_geo_params):  # may need changing...
                if order[i] >= 0:
                    # get legendre
                    p = p_list[order[i]]
                    poly = sym.S.Zero
                    for k in range(order[i] + 1):
                        if abs(pk := p[k]) > 1e-10:
                            # shift the polynomials to (a, b)
                            poly += pk * shift(mu_i) ** k
                    if poly != 0:
                        sym_mls_funcs[n] *= poly
                else:
                    sym_mls_funcs[n] *= mu_i ** order[i]
            # handel total
            # may need changing...
            if order[-1] != 0:
                sym_mls_funcs[n] *= k_term ** order[-1]

        self.sym_mls_funcs = sym.Matrix(sym_mls_funcs[np.argsort(np.sum(np.abs(mls_orders), axis=1))]).T
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

    def _compute_free_and_expanded_edges(self):
        # find the unique dirichlet edge index
        self.dirichlet_edge_index = np.unique(self.dirichlet_edge)
        # free index is unique index minus dirichlet edge index
        self.free_index = np.setdiff1d(self.tri, self.dirichlet_edge_index)

        self.expanded_free_index = helpers.expand_index(self.free_index)
        self.expanded_dirichlet_edge_index = helpers.expand_index(self.dirichlet_edge_index)

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

    def _assemble(self, geo_params):
        self.geo_params = geo_params
        if self.element in ("linear triangle", "lt"):
            self.p, self.tri, self.edge = assembly.triangle.get_plate.getPlate(self._n)
            self.a1_full, self.a2_full, self.f0_full \
                = assembly.triangle.linear.assemble_ints_and_f_body_force(self._n, self.p, self.tri,
                                                                          self.z_mat_funcs, self.geo_params,
                                                                          self.f_func, self.f_func_non_zero,
                                                                          self.nq)
            if self.has_non_homo_neumann:
                self.f0_full += assembly.triangle.linear.assemble_f_neumann(self._n, self.p, self.neumann_edge,
                                                                            self.neumann_bc_func, self.nq)
        else:
            self.p, self.tri, self.edge = assembly.quadrilateral.get_plate.getPlate(self._n)
            self.a1_full, self.a2_full, self.f0_full \
                = assembly.quadrilateral.bilinear.assemble_ints_and_f_body_force(self._n, self.p, self.tri,
                                                                                 self.z_mat_funcs, self.geo_params,
                                                                                 self.f_func, self.f_func_non_zero,
                                                                                 self.nq, self.nq_y)
            if self.has_non_homo_neumann:
                self.f0_full += assembly.quadrilateral.bilinear.assemble_f_neumann(self._n, self.p, self.neumann_edge,
                                                                                   self.neumann_bc_func, self.nq)

        self.a1_full = self.a1_full.tocsr()
        self.a2_full = self.a2_full.tocsr()

        self._edges()
        self._compute_free_and_expanded_edges()
        self.n_free = self.expanded_free_index.shape[0]

        self._set_free()
        if self.has_non_homo_dirichlet:
            self._set_f_load_dirichlet()
        self.is_assembled_and_free = True

    def assemble(self, mu1: float, mu2: float, mu3: float, mu4: float, mu5: float, mu6: float):
        self._assemble(np.array([mu1, mu2, mu3, mu4, mu5, mu6]))

    def hfsolve(self, e_young: float, nu_poisson: float, *geo_params: Optional[float], print_info: bool = True,
                drop: Union[Iterable[int], int] = None):
        if len(geo_params) != 0:
            if len(geo_params) != len(self.sym_geo_params):
                raise ValueError(
                    f"To many geometry parameters, got {len(geo_params)} expected {len(self.sym_geo_params)}.")
            if not self.mls_has_been_setup:
                raise ValueError("Matrix LSQ data functions have not been setup, please call matrix_lsq_setup.")
            if not self._mls_is_computed:
                raise ValueError("Matrices and vectors are not computed, by Matrix LSQ.")

            data = self.mls_funcs(*geo_params)
            a1_fit = mls_compute_from_fit(data, self._mls.a1_list, drop=drop)
            a2_fit = mls_compute_from_fit(data, self._mls.a2_list, drop=drop)
            f_load = mls_compute_from_fit(data, self._mls.f0_list, drop=drop)
            a = helpers.compute_a(e_young, nu_poisson, a1_fit, a2_fit)
            if self.has_non_homo_dirichlet:
                f1_dir_fit = mls_compute_from_fit(data, self._mls.f1_dir_list, drop=drop)
                f2_dir_fit = mls_compute_from_fit(data, self._mls.f2_dir_list, drop=drop)
                f_load -= helpers.compute_a(e_young, nu_poisson, f1_dir_fit, f2_dir_fit)
        elif self.is_assembled_and_free:
            # compute a
            a = helpers.compute_a(e_young, nu_poisson, self.a1, self.a2)
            # copy the body force load vector
            f_load = self.f0.copy()
            # compute and add the dirichlet load vector if it exists
            if self.has_non_homo_dirichlet:
                f_load -= helpers.compute_a(e_young, nu_poisson, self.f1_dir, self.f2_dir)
        elif self.is_assembled_and_free_from_root:
            raise ValueError("Matrices and vectors are assembled from saved data, geo_params must therefor be given.")
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
                  "The property uh, extra properties values, x and y are available.\n")

    def rbsolve(self, e_young: float, nu_poisson: float, *geo_params: float, n_rom: Optional[int] = None,
                print_info: bool = True):
        # for now
        if len(geo_params) != len(self.sym_geo_params):
            raise ValueError(
                f"To many geometry parameters, got {len(geo_params)} expected {len(self.sym_geo_params)}.")
        if not self.mls_has_been_setup:
            raise ValueError("Matrix LSQ data functions have not been setup, please call matrix_lsq_setup.")
        if not self._mls_is_computed:
            raise ValueError("Matrices and vectors are not computed, by Matrix LSQ.")
        if not self._pod_is_computed:
            raise ValueError("Pod is not computed. Can not solve.")

        if (n_rom is None) or (self._pod.n_rom == n_rom):
            # for now, may be changed
            data = self.mls_funcs(*geo_params)
            a1_fit_rom = mls_compute_from_fit(data, self.a1_rom_list)
            a2_fit_rom = mls_compute_from_fit(data, self.a2_rom_list)
            f_load_rom = mls_compute_from_fit(data, self.f0_rom_list)
            a_rom = helpers.compute_a(e_young, nu_poisson, a1_fit_rom, a2_fit_rom)
            if self.has_non_homo_dirichlet:
                f1_dir_fit_rom = mls_compute_from_fit(data, self.f1_dir_rom_list)
                f2_dir_fit_rom = mls_compute_from_fit(data, self.f2_dir_rom_list)
                f_load_rom -= helpers.compute_a(e_young, nu_poisson, f1_dir_fit_rom, f2_dir_fit_rom)
        else:
            # for now, may be changed
            if (self._last_n_rom != n_rom) or (self._a1_set_n_rom_list is None):
                self._a1_set_n_rom_list = [self._pod.compute_rom(obj, n_rom=n_rom) for obj in self._mls.a1_list]
                self._a2_set_n_rom_list = [self._pod.compute_rom(obj, n_rom=n_rom) for obj in self._mls.a2_list]
                self._f0_set_n_rom_list = [self._pod.compute_rom(obj, n_rom=n_rom) for obj in self._mls.f0_list]

            data = self.mls_funcs(*geo_params)
            a1_fit_rom = mls_compute_from_fit(data, self._a1_set_n_rom_list)
            a2_fit_rom = mls_compute_from_fit(data, self._a2_set_n_rom_list)
            f_load_rom = mls_compute_from_fit(data, self._f0_set_n_rom_list)
            a_rom = helpers.compute_a(e_young, nu_poisson, a1_fit_rom, a2_fit_rom)
            if self.has_non_homo_dirichlet:
                if (self._last_n_rom != n_rom) or (self._f1_dir_set_n_rom_list is None):
                    self._f1_dir_set_n_rom_list = [self._pod.compute_rom(obj, n_rom=n_rom)
                                                   for obj in self._mls.f1_dir_list]
                    self._f2_dir_set_n_rom_list = [self._pod.compute_rom(obj, n_rom=n_rom)
                                                   for obj in self._mls.f2_dir_list]

                f1_dir_fit_rom = mls_compute_from_fit(data, self._f1_dir_set_n_rom_list)
                f2_dir_fit_rom = mls_compute_from_fit(data, self._f2_dir_set_n_rom_list)
                f_load_rom -= helpers.compute_a(e_young, nu_poisson, f1_dir_fit_rom, f2_dir_fit_rom)
        # set last n_rom
        self._last_n_rom = n_rom
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
                  "The property uh_rom, extra properties values, x and y are available.\n")

    def hferror(self, root: Path, e_young: float, nu_poisson: float, *geo_params: Optional[float],
                drop: Union[Iterable[int], int] = None) -> float:
        # get relative HF error between true HF and MatrixLSQ HF systems
        hf_err_comp = HfErrorComputer(root)
        return hf_err_comp(self, e_young, nu_poisson, *geo_params, drop=drop)

    def rberror(self, root: Path, e_young: float, nu_poisson: float, *geo_params: float,
                n_rom: Optional[int] = None) -> float:
        # get relative RB error between true HF and RB systems
        rb_err_comp = RbErrorComputer(root)
        return rb_err_comp(self, e_young, nu_poisson, *geo_params, n_rom=n_rom)

    def save_snapshots(self, root: Path, geo_grid: int,
                       geo_range: Tuple[float, float] = None,
                       mode: str = "uniform",
                       material_grid: Optional[int] = None,
                       e_young_range: Optional[Tuple[float, float]] = None,
                       nu_poisson_range: Optional[Tuple[float, float]] = None):
        self.saver_root = root
        if geo_range is not None:
            self.geo_param_range = geo_range
        if not self.mls_has_been_setup:
            raise ValueError("Matrix LSQ data functions have not been setup, please call matrix_lsq_setup.")
        saver = SnapshotSaver(self.saver_root, geo_grid, self.geo_param_range,
                              mode=mode,
                              material_grid=material_grid,
                              e_young_range=e_young_range,
                              nu_poisson_range=nu_poisson_range)
        saver(self)

    def matrix_lsq_setup(self, mls_order: Optional[int] = 1, use_experimental_expansion: bool = False):
        self._use_experimental_expansion_in_mls = use_experimental_expansion
        # default mls_order is 1,
        assert self.mls_order >= 0
        if self.mls_order != mls_order:
            if mls_order != 1:
                assert mls_order >= 0
                self.mls_order = mls_order
            # else just use set self.mls_order

        self._sym_mls_params_setup()
        self.mls_has_been_setup = True

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
        self.a1_rom_list = [self._pod.compute_rom(obj) for obj in self._mls.a1_list]
        self.a2_rom_list = [self._pod.compute_rom(obj) for obj in self._mls.a2_list]
        self.f0_rom_list = [self._pod.compute_rom(obj) for obj in self._mls.f0_list]
        if self.has_non_homo_dirichlet:
            self.f1_dir_rom_list = [self._pod.compute_rom(obj) for obj in self._mls.f1_dir_list]
            self.f2_dir_rom_list = [self._pod.compute_rom(obj) for obj in self._mls.f2_dir_list]

    def plot_pod_singular_values(self):
        if not self._pod_is_computed:
            raise ValueError("Pod is not computed.")
        self._pod.plot_singular_values()

    def plot_pod_relative_information_content(self):
        if not self._pod_is_computed:
            raise ValueError("Pod is not computed.")
        self._pod.plot_relative_information_content()

    def get_u_exact(self, u_exact_func):
        return helpers.get_u_exact(self.p, lambda x, y: u_exact_func(*self.phi(x, y, *self.geo_params)))

    def get_geo_param_limit_estimate(self, num, round_decimal=6):
        # 1 / det(jac) has gives a rational of form 1/(k + c*x)
        # Which has the Taylor expansion
        # 1 / (k + c*x) = 1/k - cx/k^2 + c^2x^2/k^3 - ... + ...
        # for |x|<|k/c|
        # now max|x| = 1, so we need |c| < |k|
        # get k from det(jac) by setting x1=x2=0
        k = self.sym_det_jac.subs({x1: 0, x2: 0})
        k_func = sym_Lambdify(self.sym_geo_params, k)
        # get c from det(jac) via coeff and assuming x1=x2=1
        c = self.sym_det_jac.coeff(x1) + self.sym_det_jac.coeff(x2)
        cx = self.sym_det_jac - k
        c_func = sym_Lambdify(self.sym_geo_params, c)
        cx_func = sym_Lambdify(self.sym_params, cx)
        check_vec = np.linspace(0, 2 / len(self.sym_geo_params), num)
        max_geo_params = 0.0
        ant = len(self.sym_geo_params)
        for geo_params in tqdm.tqdm(product(*repeat(check_vec, ant)), desc="Iterating"):
            if abs(c_func(*geo_params)) < abs(k_func(*geo_params)):
                pot_max = max(geo_params)
                if (abs(c_func(*repeat(pot_max, ant))) < abs(k_func(*repeat(pot_max, ant)))) or \
                        (abs(c_func(*repeat(-pot_max, ant))) < abs(k_func(*repeat(-pot_max, ant)))):
                    if pot_max > max_geo_params:
                        x_vec = np.linspace(-1, 1, 101)
                        update_max = True
                        for x, y in product(x_vec, x_vec):
                            if not abs(cx_func(x, y, *repeat(pot_max, ant))) < abs(k_func(*repeat(pot_max, ant))):
                                update_max = False
                                break
                            if not abs(cx_func(x, y, *repeat(-pot_max, ant))) < abs(k_func(*repeat(-pot_max, ant))):
                                update_max = False
                                break
                        if update_max:
                            max_geo_params = pot_max

        max_geo_params = round(max_geo_params, round_decimal)
        print(f"The estimate limit for the parameters is ({-max_geo_params}, {max_geo_params}).")

    @property
    def mls_num_kept(self) -> int:
        if not self._mls_is_computed:
            raise ValueError("Matrix least square is not computed.")
        return self._mls.num_kept

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
                "High fidelity Problem has not been solved, can not return uh_free.")
        return self.uh.flatt_values[self.expanded_free_index]

    @property
    def uh_full(self) -> np.ndarray:
        if self.uh.values is None:
            raise ValueError(
                "High fidelity Problem has not been solved, can not return uh_full.")
        return self.uh.flatt_values

    @property
    def uh_anorm2(self) -> np.ndarray:
        if self.uh.values is None:
            raise ValueError(
                "High fidelity Problem has not been solved, can not return uh_anorm2.")
        return self.uh.flatt_values.T @ helpers.compute_a(self.uh.e_young, self.uh.nu_poisson, self.a1_full,
                                                          self.a2_full) @ self.uh.flatt_values

    @property
    def uh_rom_free(self) -> np.ndarray:
        if self.uh_rom.values is None:
            raise ValueError(
                "Reduced Order Problem has not been solved, can not return uh_free.")
        return self.uh_rom.flatt_values[self.expanded_free_index]

    @property
    def uh_rom_full(self) -> np.ndarray:
        if self.uh_rom.values is None:
            raise ValueError(
                "Reduced Order Problem has not been solved, can not return uh_full.")
        return self.uh_rom.flatt_values


class DraggableCornerRectangleSolver(QuadrilateralSolver):
    sym_phi = sym.expand(QuadrilateralSolver.sym_phi.subs({mu1: 0, mu2: 0,
                                                           mu3: mu1, mu4: mu2,
                                                           mu5: 0, mu6: 0}))
    sym_params = sym.Matrix([x1, x2, mu1, mu2])
    geo_param_range = (-0.49, 0.49)  # < 1/2

    @staticmethod
    def mu_to_vertices_dict():
        mu_to_vertices_dict = {"mu1": "a3 - 1", "mu2": "b3 - 1"}
        print("(x0, y0): are the coordinates of the lower left corner, default (0,0)")
        print("Given the \"Rectangle\" centered in (0, 0) with vertices in (0, 0), (0, 1), (a3, b3) and (0, "
              "1) the parameters mu1:2 are given as:")
        print(mu_to_vertices_dict)

    def __init__(self, n: int, f_func: Union[Callable, int],
                 dirichlet_bc_func: Optional[Callable] = None, get_dirichlet_edge_func: Optional[Callable] = None,
                 neumann_bc_func: Optional[Callable] = None,
                 element: str = "bq", x0: float = 0, y0: float = 0):
        super().__init__(n, f_func, dirichlet_bc_func, get_dirichlet_edge_func, neumann_bc_func, element, x0, y0)

    @classmethod
    def from_root(cls, root: Path):
        out = cls(2, 0)
        out.saver_root = root
        out._from_root()
        return out

    def assemble(self, mu1: float, mu2: float, **kwargs: float):
        self._assemble(np.array([mu1, mu2]))


class ScalableRectangleSolver(QuadrilateralSolver):
    sym_phi = sym.expand(QuadrilateralSolver.sym_phi.subs({mu1: lx - 1, mu2: 0,
                                                           mu3: lx - 1, mu4: ly - 1,
                                                           mu5: 0, mu6: ly - 1}))
    sym_params = sym.Matrix([x1, x2, lx, ly])
    geo_param_range = (0.1, 5.1)

    @staticmethod
    def mu_to_vertices_dict():
        mu_to_vertices_dict = {"lx": "Lx", "ly": "Ly"}
        print("(x0, y0): are the coordinates of the lower left corner, default (0,0)")
        print("Given the \"Rectangle\" centered in (0,0) with vertices in (0, 0), (0, Lx), (Lx, Ly) and (0, "
              "Ly) the parameters lx and ly are given as:")
        print(mu_to_vertices_dict)

    def __init__(self, n: int, f_func: Union[Callable, int],
                 dirichlet_bc_func: Optional[Callable] = None, get_dirichlet_edge_func: Optional[Callable] = None,
                 neumann_bc_func: Optional[Callable] = None,
                 element: str = "bq", x0: float = 0, y0: float = 0):
        super().__init__(n, f_func, dirichlet_bc_func, get_dirichlet_edge_func, neumann_bc_func, element, x0, y0)

    @classmethod
    def from_root(cls, root: Path):
        out = cls(2, 0)
        out.saver_root = root
        out._from_root()
        return out

    def assemble(self, lx: float, ly: float, **kwargs: float):
        self._assemble(np.array([lx, ly]))

    def get_geo_param_limit_estimate(self, **kwargs):
        print("The limit for the parameters is (0, inf).")
