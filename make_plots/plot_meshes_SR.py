# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np
import matplotlib.pyplot as plt

from fem_quadrilateral import ScalableRectangleSolver

def main():
    n = 1
    s = ScalableRectangleSolver(n, 0)
    lx, ly = 0.1, 5.1
    s.plot_mesh(lx, ly)
    plt.show()
    lx, ly = 5.1, 0.1
    s.plot_mesh(lx, ly)
    plt.show()


if __name__ == '__main__':
    main()
