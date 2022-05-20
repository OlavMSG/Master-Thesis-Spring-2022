# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np
import matplotlib.pyplot as plt

from fem_quadrilateral import DraggableCornerRectangleSolver

def main():
    n = 10
    d = DraggableCornerRectangleSolver(n, 0)
    mu1, mu2 = -0.5, -0.5
    d.plot_mesh(mu1, mu2)
    plt.show()
    mu1, mu2 = 0.5, 0.5
    d.plot_mesh(mu1, mu2)
    plt.show()
    mu1, mu2 = -0.3, -0.3
    d.plot_mesh(mu1, mu2)
    plt.show()
    mu1, mu2 = 0.3, 0.3
    d.plot_mesh(mu1, mu2)
    plt.show()


if __name__ == '__main__':
    main()
