# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import numpy as np
from time import perf_counter
from scipy.stats.qmc import LatinHypercube


def main():
    """for n in range(1, 50):
        print(n)
        rawdata = np.array([np.random.random((n,)) for _ in range(20)])
        sqrdata = rawdata.T @ rawdata
        print(sqrdata.shape)
        s = perf_counter()
        a = np.linalg.inv(sqrdata) @ rawdata.T
        print("inv time:", perf_counter() - s)
        s = perf_counter()
        b = np.linalg.solve(sqrdata, rawdata.T)
        print("solve time:", perf_counter() - s)

        print(np.allclose(a, b))"""
    n = 1_000
    a = -0.1
    b = 0.1
    sampler = LatinHypercube(d=6)
    sample = (b - a) * sampler.random(n=n) + a
    print(sample)

if __name__ == '__main__':
    main()
