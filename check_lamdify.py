# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import numpy as np
import sympy as sym
from sympy.abc import x, y



def main():
    sym_f = x ** 2 + y ** 2
    sym_f_grad = [sym_f.diff(z) for z in [x, y]]
    print(sym_f)
    print(sym_f_grad)

    f = sym.lambdify([x, y], sym_f, "numpy")
    print(type(f))
    print(f(0, 0))
    f2 = lambda x, y: x ** 2 + y ** 2
    print(type(f2))
    print(f2(0, 0))



if __name__ == '__main__':
    main()
