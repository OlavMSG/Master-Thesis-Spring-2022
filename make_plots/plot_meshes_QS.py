# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np
import matplotlib.pyplot as plt

from fem_quadrilateral import QuadrilateralSolver

def main():
    n = 80
    q = QuadrilateralSolver(n, 0)
    mu1, mu2, mu3, mu4, mu5, mu6 = -0.16, -0.16, 0.16, 0.16, -0.16, -0.16
    q.plot_mesh(mu1, mu2, mu3, mu4, mu5, mu6)
    plt.show()
    mu1, mu2, mu3, mu4, mu5, mu6 = 0.16, 0.16, -0.16, -0.16, 0.16, 0.16
    q.plot_mesh(mu1, mu2, mu3, mu4, mu5, mu6)
    plt.show()
    mu1, mu2, mu3, mu4, mu5, mu6 = -0.1, -0.1, 0.1, 0.1, -0.1, -0.1
    q.plot_mesh(mu1, mu2, mu3, mu4, mu5, mu6)
    plt.show()
    mu1, mu2, mu3, mu4, mu5, mu6 = 0.1, 0.1, -0.1, -0.1, 0.1, 0.1
    q.plot_mesh(mu1, mu2, mu3, mu4, mu5, mu6)
    plt.show()


if __name__ == '__main__':
    main()
