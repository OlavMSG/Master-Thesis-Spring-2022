# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from importlib.util import find_spec
import helpers

import numpy as np
from time import perf_counter

from assembly import triangle, quadrilateral

symengine_is_found = (find_spec("symengine") is not None)
if symengine_is_found:
    import symengine as sym
    from symengine import Lambdify as sym_lambdify
else:
    import sympy as sym
    from sympy import lambdify as sym_lambdify

x1, x2 = sym.symbols("x1, x2", real=True)
sym_x_vec = sym.Matrix([x1, x2])
a1, a2, a3, a4 = sym.symbols("a1:5", real=True)
b1, b2, b3, b4 = sym.symbols("b1:5", real=True)
mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8 = sym.symbols("mu1:9", real=True)


def sym_kron_product2x2(mat1, mat2):
    kron = sym.Matrix(
        [[mat1[0, 0] * mat2[0, 0], mat1[0, 0] * mat2[0, 1], mat1[0, 1] * mat2[0, 0], mat1[0, 1] * mat2[0, 1]],
         [mat1[0, 0] * mat2[1, 0], mat1[0, 0] * mat2[1, 1], mat1[0, 1] * mat2[1, 0], mat1[0, 1] * mat2[1, 1]],
         [mat1[1, 0] * mat2[0, 0], mat1[1, 0] * mat2[0, 1], mat1[1, 1] * mat2[0, 0], mat1[1, 1] * mat2[0, 1]],
         [mat1[1, 0] * mat2[1, 0], mat1[1, 0] * mat2[1, 1], mat1[1, 1] * mat2[1, 0], mat1[1, 1] * mat2[1, 1]]]
    )
    return kron


class QuadrilateralSolver:
    ref_plate = (0, 1)
    implemented_elements = ["linear triangle", "lt", "bilinear quadrilateral", "bq"]
    sym_phi = sym.Matrix([
        x1 + mu1 * (1 - x1) * (1 - x2) + mu3 * x1 * (1 - x2) + mu5 * x1 * x2 + mu7 * (1 - x1) * x2,
        x2 + mu2 * (1 - x1) * (1 - x2) + mu4 * x1 * (1 - x2) + mu6 * x1 * x2 + mu8 * (1 - x1) * x2
    ])
    sym_params = sym.Matrix([x1, x2, mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8])
    """mu_to_vertices_dict = {mu1: a1, mu2: b1,
                           mu3: a2 - 1, mu4: b2,
                           mu5: a3 - 1, mu6: b3 - 1,
                           mu7: a4, mu8: b4 - 1}"""

    def _sym_setup(self):
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
                self.z_mat_funcs[i, j] = np.vectorize(sym_lambdify(self.sym_params, self.sym_z_mat[i, j]),
                                                      otypes=[float])
        print("func time:", perf_counter() - s)

    def _sym_mls_params_setup(self):
        s = perf_counter()
        self.sym_geo_params = self.sym_params[2:]

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
        self.mls_funcs = sym_lambdify(self.sym_params, self.sym_mls_funcs)
        print("mls params:", perf_counter() - s)
        print(self.sym_mls_funcs)

    def __init__(self, n, element="lt"):

        self.mls_order = None
        self.use_negative_mls_order = None
        self.n = n + 1

        self._sym_setup()

        def default_func(x, y):
            return 0, 0

        self.f_func = helpers.VectorizedFunction2D(default_func)
        self.f_func_non_zero = False

        if element.lower() in self.implemented_elements:
            self.element = element.lower()
        else:
            error_text = "Element " + str(element) + " is not implemented. " \
                         + "Implemented elements: " + str(self.implemented_elements)
            raise NotImplementedError(error_text)

        if self.element in ("linear triangle", "lt"):
            self.nq = 4
        else:
            self.nq = 2
            self.nq_y = 2

    def mls(self, mls_order=1, use_negative_mls_order=False):
        self.mls_order = mls_order
        self.use_negative_mls_order = use_negative_mls_order

        self._sym_mls_params_setup()

    def set_quadrature_scheme_order(self, nq, nq_y=None):
        if self.element in ("linear triangle", "lt"):
            self.nq = nq
        else:
            self.nq = nq
            if nq_y is None:
                self.nq_y = nq
            else:
                self.nq_y = nq

    def _assemble(self, geo_params):
        s = perf_counter()
        self.geo_params = geo_params
        if self.element in ("linear triangle", "lt"):
            self.p, self.tri, self.edge = triangle.get_plate.getPlate(self.n)
            triangle.linear.assemble_ints_and_f_load_lv(self.n, self.p, self.tri, self.z_mat_funcs, self.geo_params,
                                                        self.f_func, self.f_func_non_zero, self.nq)
        else:
            self.p, self.tri, self.edge = quadrilateral.get_plate.getPlate(self.n)
            quadrilateral.bilinear.assemble_ints_and_f_load_lv(self.n, self.p, self.tri, self.z_mat_funcs,
                                                               self.geo_params, self.f_func, self.f_func_non_zero,
                                                               self.nq, self.nq_y)
        print("time assemble:", perf_counter() - s)

    def assemble(self, mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, **kwargs):
        self._assemble(np.array([mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8]))


class DraggableCornerRectangleSolver(QuadrilateralSolver):
    sym_phi = sym.expand(QuadrilateralSolver.sym_phi.subs({mu1: 0, mu2: 0,
                                                           mu3: 0, mu4: 0,
                                                           mu5: mu1, mu6: mu2,
                                                           mu7: 0, mu8: 0}))
    sym_params = sym.Matrix([x1, x2, mu1, mu2])

    def __init__(self, n):
        super().__init__(n)

    def assemble(self, mu1, mu2, **kwargs):
        self._assemble(np.array([mu1, mu2]))


class ScalableRectangleSolver(QuadrilateralSolver):
    sym_phi = sym.expand(QuadrilateralSolver.sym_phi.subs({mu1: 0, mu2: 0,
                                                           mu3: mu1 - 1, mu4: 0,
                                                           mu5: mu1 - 1, mu6: mu2 - 1,
                                                           mu7: 0, mu8: mu2 - 1}))
    sym_params = sym.Matrix([x1, x2, mu1, mu2])

    def __init__(self, n):
        super().__init__(n)

    def assemble(self, mu1, mu2, **kwargs):
        self._assemble(np.array([mu1, mu2]))


def main():
    n = 3
    # q = QuadrilateralSolver(n)
    # print(q.sym_phi)
    d = DraggableCornerRectangleSolver(n)
    print(d.sym_phi)
    d.assemble(0.1, 0.2)
    print("------------")
    r = ScalableRectangleSolver(n)
    print(r.sym_phi)
    r.assemble(1, 2)
    """u = set([item for sublist in r.sym_z_mat.tolist() for item in sublist])
    print(u)"""


if __name__ == '__main__':
    main()
