# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
based on Specialization-Project-fall-2021
"""
import os

import numpy as np


def index_map(i, d):
    """
    The index map used mapping the from 2D index to 1D index

    Parameters
    ----------
    i : int, np.array
        the index in the 2D case.
    d : int, np.array
        the dimension to use for the 2D index.

    Returns
    -------
    int, np.array
        1D index.

    """
    return 2 * i + d


def inv_index_map(k):
    """
    The inverse index map used mapping the from 1D index to 2D index

    Parameters
    ----------
    k : int
        1D index.

    Returns
    -------
    i : int
        the index in the 2D case.
    d : int
        the dimension to use for the 2D index.

    """
    return divmod(k, 2)


def expand_index(index):
    """
    Expand an array of 2D indexes to the corresponding array of 1D indexes

    Parameters
    ----------
    index : np.array
        array of 2D indexes.

    Returns
    -------
    expanded_index : np.array
        corresponding array of 1D indexes.

    """
    m = index.shape[0] * 2
    expanded_index = np.zeros(m, dtype=int)
    expanded_index[np.arange(0, m, 2)] = index_map(index, 0)
    expanded_index[np.arange(1, m, 2)] = index_map(index, 1)
    return expanded_index


def get_lambda_mu(e_young, nu_poisson):
    """
    Get 2D plane stress Lame coefficients lambda and mu from the young's module and the poisson ratio

    Parameters
    ----------
    e_young : float, np.float, np.ndarray
        young's module.
    nu_poisson : float, np.float, np.ndarray
        poisson ratio.

    Returns
    -------
    mu : float, np.array
        Lame coefficient mu.
    lambda_bar : float, np.array
        Lame coefficient lambda.

    """
    if e_young == 0:
        raise ValueError("Invalid Young's module; E=0.")
    if abs(nu_poisson) >= 1:
        raise ValueError("Invalid Poisson Ratio; |nu|>=1.")
    lambda_bar = e_young * nu_poisson / (1 - nu_poisson * nu_poisson)
    mu = 0.5 * e_young / (nu_poisson + 1)
    return mu, lambda_bar


def get_e_young_nu_poisson(mu, lambda_bar):
    """
    Get the young's module and the poisson ratio from 2D plane stress Lame coefficients lambda and mu
    (Note: used formulas in get_lambda_mu and solved for e_young and nu_poisson)

    Parameters
    ----------
    mu : float, np.float
        Lame coefficient mu.
    lambda_bar : float, np.float
        Lame coefficient lambda.

    Returns
    -------
    e_young : float
        young's module.
    nu_poisson : float
        poisson ratio.

    """
    nu_poisson = lambda_bar / (lambda_bar + 2 * mu)
    e_young = 4 * (lambda_bar * mu + mu * mu) / (lambda_bar + 2 * mu)
    return e_young, nu_poisson


def compute_a(e_young, nu_poisson, a1, a2):
    """
    Compute the matrix a from the linear elasticity problem,
    depending on the young's module and the poisson ratio,
    and the matrices a1 and a2

    Parameters
    ----------
    e_young : float, np.float
        young's module.
    nu_poisson : float, np.float
        poisson ratio.
    a1 : scipy.sparse.dox_matrix, np.array
        bilinar form matrix a1.
    a2 : scipy.sparse.dox_matrix, np.array
        bilinar form matrix a2.

    Returns
    -------
    scipy.sparse.dox_matrix, np.array
        bilinar form matrix a depending on the young's module and the poisson ratio.

    """
    # get the Lame coefficients
    mu, lambda_bar = get_lambda_mu(e_young, nu_poisson)
    # compute a
    return 2 * mu * a1 + lambda_bar * a2


def compute_a1_and_a2(int1, int2, int3, int4, int5):
    """
    Compute the matrices a1 and a2 from the 5 integrals

    Parameters
    ----------
    int1 : scipy.sparse.dox_matrix, np.array
        bilinar form matrix int1.
    int2 : scipy.sparse.dox_matrix, np.array
        bilinar form matrix int2.
    int3 : scipy.sparse.dox_matrix, np.array
        bilinar form matrix int3.
    int4 : scipy.sparse.dox_matrix, np.array
        bilinar form matrix int4.
    int5 : scipy.sparse.dox_matrix, np.array
        bilinar form matrix int5.


    Returns
    -------
    a1 : scipy.sparse.dox_matrix, np.array
        bilinar form matrix a1.
    a2 : scipy.sparse.dox_matrix, np.array
        bilinar form matrix a2.
    """
    a1 = int1 + 0.5 * (int2 + int3 + int4)
    a2 = int1 + int3 + int5
    return a1, a2


def get_u_exact(p, u_exact_func):
    """
    Get a FunctionValues2D representation of the exact solution

    Parameters
    ----------
    p : np.array
        Nodal points, (x,y)-coordinates for point i given in row i.
    u_exact_func : function
        function representing the exact solution of the problem.

    Returns
    -------
    np.array
        the FunctionValues2D representation of the exact solution, form [x1, y1, x2, y2, ...].

    """
    x_vec = p[:, 0]
    y_vec = p[:, 1]

    u_exact = FunctionValues2D.from_2xn(VectorizedFunction2D(u_exact_func)(x_vec, y_vec))
    return u_exact


def check_and_make_folder(n, folder_path, n_counts_nodes=False):
    """
    Check if the folder/directory and its sub-folder exists, if not make it.
    Check both the folders 'folder_path' and 'folder_path/_n{_n}'

    Parameters
    ----------
    n : int
        Discretization number
    folder_path : str
        the path to the folder to check and make, form 'folder_path/_n{_n}'.
    n_counts_nodes: bool
        True if _n counts the number of nodes along the axes. Default False

    Returns
    -------
    str
        the folder name .
    """
    if n_counts_nodes:
        m = n - 1
    else:
        m = n
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    folder_path = os.path.join(folder_path, f"_n{m}")
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    return folder_path


def get_vec_from_range(range_, m, mode):
    """
    Get vector of m uniform or Gauss-Lobatto points from range_
    Parameters
    ----------
    range_ : tuple
        the range of numbers to consider.
    m : int
        number of points in vector.
    mode : str
        sampling sample_mode, uniform or Gauss-Lobatto.
    Raises
    ------
    NotImplementedError
        if sample_mode is not uniform or Gauss-Lobatto.
    Returns
    -------
    np.array
        array of sampling points.
    """
    if mode.lower() == "uniform":
        return np.linspace(range_[0], range_[1], m)

    elif mode.lower() == "gauss-lobatto":
        from quadpy.c1 import gauss_lobatto
        return 0.5 * ((range_[1] - range_[0]) * gauss_lobatto(m).points + (range_[1] + range_[0]))
    else:
        raise NotImplementedError(
            f"Mode {mode} is not implemented. The implemented modes are uniform and gauss lobatto.")


# class to vectorized input functions
class VectorizedFunction2D:

    def __init__(self, func_non_vec):
        """
        Set up to vectorized a function with 2D input and output

        Parameters
        ----------
        func_non_vec : function
            function to vectorize.

        Returns
        -------
        None.

        """

        def vectorize_func_2d(x_vec, y_vec):
            """
            Vectorize a function with 2D input and output

            Parameters
            ----------
            x_vec : np.array
                array of x-point.
            y_vec : np.array
                array of y-point.

            Returns
            -------
            np.array
                matrix, row 0: x-values, row 1: y-values.

            """
            if isinstance(x_vec, (float, int)):
                x_vec = np.array([x_vec])
            if isinstance(y_vec, (float, int)):
                y_vec = np.array([y_vec])
            x_vals = np.zeros_like(x_vec, dtype=float)
            y_vals = np.zeros_like(x_vec, dtype=float)
            for i, (x, y) in enumerate(zip(x_vec, y_vec)):
                x_vals[i], y_vals[i] = func_non_vec(x, y)
            return np.row_stack((x_vals, y_vals))

        self._func_vec = vectorize_func_2d

    def __call__(self, x_vec, y_vec):
        """
        Vectorize a function with 2D input and output

        Parameters
        ----------
        x_vec : np.array
            array of x-point.
        y_vec : np.array
            array of y-point.

        Returns
        -------
        np.array
            matrix, row 0: x-values, row 1: y-values.

        """
        return self._func_vec(x_vec, y_vec)


class FunctionValues2D:

    def __init__(self):
        """
        Setup

        Returns
        -------
        None.

        """
        self._values = None
        self._n = None

    def __repr__(self):
        return self._values.__repr__()

    def __str__(self):
        return self._values.__str__()

    def _set_from_nx2(self, values):
        """
        set from values of shape (_n,2)

        Parameters
        ----------
        values : np.array
            function values in shape (_n,2).

        Returns
        -------
        None.

        """

        self._values = np.asarray(values, dtype=float)
        self._n = self._values.shape[0]

    def _set_from_2xn(self, values):
        """
        set from values of shape (2,_n)

        Parameters
        ----------
        values : np.array
            function values in shape (2,_n).

        Returns
        -------
        None.

        """
        self._values = np.asarray(values.T, dtype=float)
        self._n = self._values.shape[0]

    def _set_from_1x2n(self, values):
        """
        set from values of shape (1, k=2n)

        Parameters
        ----------
        values : np.array
            function values in shape (1, k=2n).

        Raises
        ------
        ValueError
            if k != 2n.

        Returns
        -------
        None.

        """
        m = values.shape[0]
        if m % 2 != 0:
            raise ValueError("Shape of values must be (1, k=2n), where _n is an integer.")
        self._n = m // 2
        self._values = np.zeros((self.n, 2))
        self._values[:, 0] = values[np.arange(0, m, 2)]
        self._values[:, 1] = values[np.arange(1, m, 2)]

    @classmethod
    def from_nx2(cls, values):
        """
        Make FunctionValues2D from values of shape (_n, 2)

        Parameters
        ----------
        values : np.array
            function values in shape (_n,2).

        Returns
        -------
        self : FunctionValues2D
            FunctionValues2D from values of shape (_n, 2).

        """
        out = cls()
        out._set_from_nx2(values)
        return out

    @classmethod
    def from_2xn(cls, values):
        """
        Make FunctionValues2D from values of shape (2, _n)

        Parameters
        ----------
        values : np.array
            function values in shape (_n,2).

        Returns
        -------
        self : FunctionValues2D
            FunctionValues2D from values of shape (2, _n).

        """
        out = cls()
        out._set_from_2xn(values)
        return out

    @classmethod
    def from_1x2n(cls, values):
        """
        Make FunctionValues2D from values of shape (1, k=2n)

        Parameters
        ----------
        values : np.array
            function values in shape (1,2n).

        Returns
        -------
        self : FunctionValues2D
            FunctionValues2D from values of shape (1, k=2n).

        """
        out = cls()
        out._set_from_1x2n(values)
        return out

    @property
    def values(self):
        """
        Values

        Returns
        -------
        np.array
            values

        """
        return self._values

    @property
    def x(self):
        """
        x-values

        Returns
        -------
        np.array
            x-values.

        """
        return self._values[:, 0]

    @property
    def y(self):
        """
        y-values

        Returns
        -------
        np.array
            y-values.

        """
        return self._values[:, 1]

    @property
    def flatt_values(self):
        """
        The flattened values

        Returns
        -------
        np.array
            flatted values in form [x0, y0, x1, y1, ...].

        """
        # return [x0, y0, x1, y1, ...]
        return self._values.reshape((self._n * 2,)).ravel()

    @property
    def dim(self):
        """
        Dimension

        Returns
        -------
        int
            dimension.

        """
        return 2

    @property
    def n(self):
        """
        Number of (x,y)-values

        Returns
        -------
        int
            number of (x,y)-values.

        """
        return self._n

    @property
    def shape(self):
        """
        Shape

        Returns
        -------
        tuple
            shape of values.

        """
        return self._values.shape