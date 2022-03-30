# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import mpmath
import numpy as np
import sympy as sym
from scipy.special import legendre
from sympy.abc import x


def main():
    from scipy.special import roots_legendre
    max_order = 10
    sym_f = sym.Matrix([sym.S.One] * (max_order + 1))
    p_list = [legendre(p).coef[::-1] for p in range(max_order + 1)]
    a = sym.legendre(2, x)
    print(a, "ewrhjiou")
    for order in range(max_order + 1):
        # get legendre
        p = p_list[order]
        poly = sym.S.Zero
        for k in range(order + 1):
            if abs(pk := p[k]) > 1e-10:
                poly += pk * x ** k
        if poly != 0:
            sym_f[order] *= poly

    print(sym_f)
    sym_1x = 1 / x
    z, rho = roots_legendre(4)
    for k in range(len(sym_f)):
        f = sym.lambdify(x, sym_f[k])
        # PV. \int_{-1}^1 dx 1/(x - wvar) * f
        pv = np.sum(rho * f(z) / z)
        print(pv, k)
        if abs(pv) > 1e-10:
            pk_pk = np.sum(rho * (f(z) * f(z)))
            if abs(pk_pk) > 1e-10:
                sym_1x -= pv / pk_pk * sym_f[k]

    print(sym_1x)
    f_1x = sym.lambdify(x, sym_1x)
    for k in range(len(sym_f)):
        v = np.sum(rho * f_1x(z))
        print(v, k)


if __name__ == '__main__':
    main()

