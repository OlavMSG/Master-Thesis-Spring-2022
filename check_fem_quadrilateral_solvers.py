# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from fem_quadrilateral import DraggableCornerRectangleSolver, ScalableRectangleSolver, QuadrilateralSolver


def main():
    n = 3
    order = 3
    q = QuadrilateralSolver(n, 0)
    print(q.sym_phi)
    """print(q.sym_jac)
    print(q.sym_det_jac)
    q.matrix_lsq_setup(mls_order=order)
    print(q.sym_mls_funcs)
    print(len(q.sym_mls_funcs))"""
    q.get_geo_param_limit_estimate(5)
    print("------------")
    d = DraggableCornerRectangleSolver(n, 0)
    print(d.sym_phi)
    """print(d.sym_jac)
    print(d.sym_det_jac)
    d.matrix_lsq_setup(mls_order=order)
    print(d.sym_mls_funcs)
    print(len(d.sym_mls_funcs))"""
    """d.matrix_lsq_setup()
    print(d.sym_mls_funcs)
    print(d.mls_funcs(.3, .5).ravel())
    # d.assemble(0.1, 0.2)"""
    d.get_geo_param_limit_estimate(101)
    print("------------")
    r = ScalableRectangleSolver(n, 0)
    print(r.sym_phi)
    """for i in range(4):
        for j in range(4):
            print(r.z_mat_funcs[i, j](np.ones(2), np.ones(2), 2, 3))"""
    """r.matrix_lsq_setup(mls_order=order)
    print(r.sym_mls_funcs)
    print(len(r.sym_mls_funcs))"""
    r.get_geo_param_limit_estimate()



if __name__ == '__main__':
    main()
