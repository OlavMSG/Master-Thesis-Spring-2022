# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from fem_quadrilateral import DraggableCornerRectangleSolver, ScalableRectangleSolver


def main():
    n = 3
    # q = QuadrilateralSolver(n, 0)
    # q.mls()
    # print(q.sym_phi)
    d = DraggableCornerRectangleSolver(n, 0)
    print(d.sym_phi)
    d.matrix_lsq_setup()
    print(d.sym_mls_funcs)
    print(d.mls_funcs(.3, .5).ravel())
    # d.assemble(0.1, 0.2)
    print("------------")
    r = ScalableRectangleSolver(n, 0)
    print(r.sym_phi)
    """for i in range(4):
        for j in range(4):
            print(r.z_mat_funcs[i, j](np.ones(2), np.ones(2), 2, 3))"""

    r.assemble(1, 2)
    r.hfsolve(200e3, 0.5)


if __name__ == '__main__':
    main()
