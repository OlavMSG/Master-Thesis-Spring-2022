# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations

from importlib.util import find_spec
from typing import Optional

from scipy.sparse.linalg import spsolve
import numpy as np
from time import perf_counter

import helpers
from assembly import triangle, quadrilateral
from solution_function_class import SolutionFunctionValues2D
from base_solver import BaseSolver

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

    @staticmethod
    def mu_to_vertices_dict():
        mu_to_vertices_dict = {"mu1": "a1", "mu2": "b1",
                               "mu3": "a2 - 1", "mu4": "b2",
                               "mu5": "a3 - 1", "mu6": "b3 - 1",
                               "mu7": "a4", "mu8": "b4 - 1"}
        print("Given the Quadrilateral with vertices in (a1, b1), (a2, b2), (a3, b3) and (a4, b4) the parameter mu1:8 "
              "are given as:")
        print(mu_to_vertices_dict)

    def __init__(self, n, f_func, dirichlet_bc_func=None, element="bq"):

        self.mls_has_been_setup = False
        self.uh = SolutionFunctionValues2D()
        self.neumann_edge = None
        self.dirichlet_edge = None
        self.get_dirichlet_edge_func = None
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

        self.f_func_non_zero = True
        if f_func == 0:
            self.f_func = helpers.VectorizedFunction2D(default_func)
            self.f_func_non_zero = False
        else:
            self.f_func = helpers.VectorizedFunction2D(f_func)

        self.has_non_homo_dirichlet = False
        if dirichlet_bc_func is None:
            self.dirichlet_bc_func = helpers.VectorizedFunction2D(default_func)
        else:
            self.dirichlet_bc_func = helpers.VectorizedFunction2D(
                lambda x, y: dirichlet_bc_func(*self.phi(x, y, *self.geo_params)))
            self.has_non_homo_dirichlet = True

    def _sym_setup(self):
        self.sym_geo_params = self.sym_params[2:]
        s = perf_counter()
        self.phi = sym_Lambdify(self.sym_params, self.sym_phi)
        print("phi time:", perf_counter() - s)
        s = perf_counter()
        self.sym_jac = self.sym_phi.jacobian(sym_x_vec)
        self.sym_det_jac = self.sym_jac.det()
        self.is_jac_constant = (x1 and x2 not in self.sym_jac.free_symbols)
        sym_jac_inv_det = sym.Matrix([[self.sym_jac[1, 1], - self.sym_jac[0, 1]],
                                      [-self.sym_jac[1, 0], self.sym_jac[0, 0]]])
        print("jac time:", perf_counter() - s)
        s = perf_counter()
        self.sym_z_mat = sym_kron_product2x2(sym_jac_inv_det, sym_jac_inv_det) / self.sym_det_jac
        print("z time:", perf_counter() - s)

        self.z_mat_funcs = np.empty((4, 4), dtype=object)
        s = perf_counter()
        for i in range(4):
            for j in range(4):
                self.z_mat_funcs[i, j] = np.vectorize(sym_Lambdify(self.sym_params, self.sym_z_mat[i, j]),
                                                      otypes=[float])
        print("func time:", perf_counter() - s)

    def _sym_mls_params_setup(self):
        s = perf_counter()

        if self.is_jac_constant:
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
        self.mls_funcs = sym_Lambdify(self.sym_params, self.sym_mls_funcs)
        print("mls params:", perf_counter() - s)
        # print(self.sym_mls_funcs)

    def mls_setup(self, mls_order: int = 1, use_negative_mls_order: bool = False):
        self.mls_order = mls_order
        self.use_negative_mls_order = use_negative_mls_order

        self._sym_mls_params_setup()
        self.mls_has_been_setup = True

    def set_quadrature_scheme_order(self, nq: int, nq_y: Optional[int] = None):
        if self.element in ("linear triangle", "lt"):
            self.nq = nq
        else:
            self.nq = nq
            if nq_y is None:
                self.nq_y = nq
            else:
                self.nq_y = nq

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

    def _get_neumann_edge(self):
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

    def _edges(self):
        """
        Get the edges
        Returns
        -------
        None.
        """
        self._get_dirichlet_edge()
        self._get_neumann_edge()
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

    def _assemble(self, geo_params):
        s = perf_counter()
        self.geo_params = geo_params
        if self.element in ("linear triangle", "lt"):
            self.p, self.tri, self.edge = triangle.get_plate.getPlate(self.n)
            self.ints, self.f_body_force_full \
                = triangle.linear.assemble_ints_and_f_body_force(self.n, self.p, self.tri, self.z_mat_funcs,
                                                                 self.geo_params, self.f_func,
                                                                 self.f_func_non_zero, self.nq)
        else:
            self.p, self.tri, self.edge = quadrilateral.get_plate.getPlate(self.n)
            self.ints, self.f_body_force_full \
                = quadrilateral.bilinear.assemble_ints_and_f_body_force(self.n, self.p, self.tri, self.z_mat_funcs,
                                                                        self.geo_params, self.f_func,
                                                                        self.f_func_non_zero, self.nq, self.nq_y)
        self.a1_full, self.a2_full = helpers.compute_a1_and_a2(*self.ints)
        # NB!!! for now
        self.f0_full = self.f_body_force_full
        print("time assemble:", perf_counter() - s)

        s = perf_counter()
        self._edges()
        self._compute_free_and_expanded_edges()
        self.n_free = self.expanded_free_index.shape[0]

        self._set_free()
        if self.has_non_homo_dirichlet:
            self._set_f_load_dirichlet()
        print("time free and lifting:", perf_counter() - s)
        self.is_assembled_and_free = True

    def _compute_a_free(self, e_young, nu_poisson):
        if self.is_assembled_and_free:
            return helpers.compute_a(e_young, nu_poisson, self.a1, self.a2)
        else:
            raise ValueError("Matrices and vectors are not assembled.")

    def compute_f_load_free(self, e_young, nu_poisson):

        # copy the body force load vector
        f_load = self.f0.copy()
        """# add the neumann load vector if it exists
        if self.has_neumann and self.has_non_homo_neumann:
            f_load += self._hf_data.f_load_neumann_free"""
        # compute and add the dirichlet load vector if it exists
        if self.has_non_homo_dirichlet:
            if e_young is None or nu_poisson is None:
                raise ValueError("e_young and/or nu_poisson are not given, needed for non homo. dirichlet conditions.")
            f_load -= helpers.compute_a(e_young, nu_poisson, self.f1_dir, self.f2_dir)
        return f_load

    def hfsolve(self, e_young: float, nu_poisson: float, print_info: bool = True):
        """
        Solve the High-fidelity system given a Young's module and poisson ratio
        Parameters
        ----------
        e_young : float, np.ndarray
            Young's module.
        nu_poisson : float, np.ndarray
            poisson ratio.
        print_info : bool, optional
            print solver info. The default is True.
        Returns
        -------
        None.
        """
        # compute a and convert to csr
        a = self._compute_a_free(e_young, nu_poisson).tocsr()
        f_load = self.compute_f_load_free(e_young, nu_poisson)  # e_young, nu_poisson needed if non homo. dirichlet
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

    def assemble(self, mu1: float, mu2: float, mu3: float, mu4: float, mu5: float, mu6: float, mu7: float, mu8: float):
        self._assemble(np.array([mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8]))

    def get_u_exact(self, u_exact_func):
        return helpers.get_u_exact(self.p, lambda x, y: u_exact_func(*self.phi(x, y, *self.geo_params)))

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
        print("Given the \"Rectangle\" with vertices in (0, 0), (0, 1), (a3, b3) and (0, 1) the parameter mu1:2 "
              "are given as:")
        print(mu_to_vertices_dict)

    def __init__(self, n, f_func, dirichlet_bc_func=None, element="bq"):
        super().__init__(n, f_func, dirichlet_bc_func, element)

    def assemble(self, mu1: float, mu2: float, **kwargs: float):
        self._assemble(np.array([mu1, mu2]))


class ScalableRectangleSolver(QuadrilateralSolver):
    sym_phi = sym.expand(QuadrilateralSolver.sym_phi.subs({mu1: 0, mu2: 0,
                                                           mu3: mu1 - 1, mu4: 0,
                                                           mu5: mu1 - 1, mu6: mu2 - 1,
                                                           mu7: 0, mu8: mu2 - 1}))
    sym_params = sym.Matrix([x1, x2, mu1, mu2])
    geo_param_range = (1, 5)

    @staticmethod
    def mu_to_vertices_dict():
        mu_to_vertices_dict = {"mu1": "Lx", "mu2": "Ly"}
        print("Given the \"Rectangle\" with vertices in (0, 0), (0, Lx), (Lx, Ly) and (0, Ly) the parameter mu1:2 "
              "are given as:")
        print(mu_to_vertices_dict)

    def __init__(self, n, f_func, dirichlet_bc_func=None, element="bq"):
        super().__init__(n, f_func, dirichlet_bc_func, element)

    def assemble(self, mu1: float, mu2: float, **kwargs: float):
        self._assemble(np.array([mu1, mu2]))


def main():
    n = 3
    # q = QuadrilateralSolver(n, 0)
    # q.mls()
    # print(q.sym_phi)
    d = DraggableCornerRectangleSolver(n, 0)
    print(d.sym_phi)
    d.mls_setup()
    d.assemble(0.1, 0.2)
    print("------------")
    r = ScalableRectangleSolver(n, 0)
    print(r.sym_phi)
    r.mls_setup()
    r.assemble(1, 2)
    """u = set([item for sublist in r.sym_z_mat.tolist() for item in sublist])
    print(u)"""


if __name__ == '__main__':
    main()
