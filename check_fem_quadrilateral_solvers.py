# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from fem_quadrilateral import DraggableCornerRectangleSolver, ScalableRectangleSolver, QuadrilateralSolver


def main():
    n = 2
    order = 3
    q = QuadrilateralSolver(n, 0)
    print(q.sym_phi)
    print(q.sym_jac)
    print(q.sym_det_jac)
    q.matrix_lsq_setup(mls_order=order)
    print(q.sym_mls_funcs)
    print(len(q.sym_mls_funcs))
    # q.get_geo_param_limit_estimate(5)
    print("------------")
    d = DraggableCornerRectangleSolver(n, 0)
    print(d.sym_phi)
    print(d.sym_jac)
    print(d.sym_det_jac)
    d.matrix_lsq_setup(mls_order=order)
    print(d.sym_mls_funcs)
    print(len(d.sym_mls_funcs))
    # d.assemble(0.1, 0.2)
    # d.get_geo_param_limit_estimate(101)
    print("------------")
    r = ScalableRectangleSolver(n, 0)
    print(r.sym_phi)
    r.matrix_lsq_setup(mls_order=order)
    print(r.sym_mls_funcs)
    print(len(r.sym_mls_funcs))
    # r.get_geo_param_limit_estimate()



if __name__ == '__main__':
    main()
