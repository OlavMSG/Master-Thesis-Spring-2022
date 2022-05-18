# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations

from typing import Union, Tuple
import numpy as np

from .default_constants import default_tol
from .helpers import FunctionValues2D


class SolutionFunctionValues2D(FunctionValues2D):

    def __init__(self):
        """
        Setup

        Returns
        -------
        None.

        """
        super().__init__()
        self._e_young = None
        self._nu_poisson = None
        self._geo_params = None
        self._nodal_stress = None
        self._von_mises = None

    def set_geo_params(self, geo_params: Union[np.ndarray, Tuple[float, ...]]):
        """
        set which geometry parameters that was used

        Parameters
        ----------
        geo_params : np.ndarray, tuple
            geometry parameters

        Returns
        -------
        None.

        """
        self._geo_params = geo_params

    def set_e_young_and_nu_poisson(self, e_young: float, nu_poisson: float):
        """
        set which young's module and poisson ratio that was uses

        Parameters
        ----------
        e_young : float
            young's module.
        nu_poisson : float
            poisson ratio.

        Returns
        -------
        None.

        """
        self._e_young = e_young
        self._nu_poisson = nu_poisson

    def check_e_young_and_nu_poisson(self, e_young: float, nu_poisson: float) -> bool:
        """
        Check if input young's module and poisson ratio is what was used

        Parameters
        ----------
        e_young : float
            young's module.
        nu_poisson : float
            poisson ratio.

        Returns
        -------
        bool
            True if input young's module and poisson ratio is what was used.

        """
        if self._values is None:
            return False
        elif abs(self._e_young - e_young) <= default_tol and abs(self._nu_poisson - nu_poisson) <= default_tol:
            return True
        else:
            return False

    def set_nodal_stress(self, nodal_stress: np.ndarray):
        """
        set the nodal stess property of the class

        Parameters
        ----------
        nodal_stress : np.ndarray
            nodal stress.

        Returns
        -------
        None.

        """
        self._nodal_stress = nodal_stress

    def set_von_mises(self, von_mises: np.ndarray):
        """
        set the von mises yield property of the class

        Parameters
        ----------
        von_mises : np.ndarray
            von mises yield.

        Returns
        -------
        None.

        """
        self._von_mises = von_mises

    @property
    def geo_params(self) -> Union[np.ndarray, Tuple[float, ...]]:
        """
        The geometry parameters

        Returns
        -------
        np.ndarray, tuple
            the geometry parameters

        """
        return self._geo_params

    @property
    def e_young(self) -> float:
        """
        The Young's module E

        Returns
        -------
        float
            the Young's module E.

        """
        return self._e_young

    @property
    def nu_poisson(self) -> float:
        """
        The Poisson ratio nu

        Returns
        -------
        float
            the Poisson ratio nu.

        """
        return self._nu_poisson

    @property
    def nodal_stress(self) -> np.ndarray:
        """
        The nodal stress

        Returns
        -------
        np.ndarray
            nodal stress.

        """
        return self._nodal_stress

    @property
    def von_mises(self) -> np.ndarray:
        """
        Von Mises stress

        Returns
        -------
        np.ndarray
            von Mises stress.

        """
        return self._von_mises
