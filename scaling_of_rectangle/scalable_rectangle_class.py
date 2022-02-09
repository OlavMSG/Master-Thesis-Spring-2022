# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import numpy as np


class ScalableRectangle:
    ref_plate = (-1, 1)
    is_jac_constant = True

    def __init__(self, lx=1, ly=1):
        self.lx = lx
        self.ly = ly

    def phi(self, x, y):
        return 0.5 * (x + self.lx), 0.5 * (y + self.ly)

    @property
    def jac(self):
        return np.array([[0.5 * self.lx, 0],
                         [0, 0.5 * self.ly]])

    @property
    def det_jac(self):
        return 0.25 * self.lx * self.ly

    @property
    def jac_inv(self):
        return np.array([[2 / self.lx, 0],
                         [0, 2 / self.ly]])

    @property
    def z(self):
        return np.array([[self.ly / self.lx, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, self.lx / self.ly]])
