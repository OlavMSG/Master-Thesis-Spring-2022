# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import numpy as np

def main():
    np.random.seed(1234)
    a = np.random.random((4, 4))
    a = a + a.T

    free_index = np.array([1, 2])
    dir_index = np.array([0, 3])
    index = np.ix_(free_index, free_index)
    a_free = a[index]
    b = np.random.random(2)
    rg = np.random.random(2)
    u_ex = np.random.random(4)
    u_ex[dir_index] = rg
    uh = np.zeros(4)
    uh[free_index] = np.linalg.solve(a_free, b)
    uh[dir_index] = rg
    print(a)

    print("u")
    print(uh)
    print(u_ex)
    print(np.abs(uh - u_ex))

    print("a-norm2")
    err1 = uh - u_ex
    print(err1 @ a @ err1)
    err2 = err1[free_index]
    print(err2 @ a_free @ err2)


if __name__ == '__main__':
    main()
