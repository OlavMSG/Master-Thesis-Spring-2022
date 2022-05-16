# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations
import numpy as np
from .helpers import get_lambda_mu
from .base_solver import BaseSolver
from . import assembly


def sigma_lt(e_young: float, nu_poisson: float, ck: np.ndarray, i: int, d: int, phi_jac_inv: np.ndarray):
    # get Lambert's coeffs.
    mu, lambda_bar = get_lambda_mu(e_young, nu_poisson)
    nabla_grad_non_ref = phi_jac_inv.T @ assembly.triangle.linear.nabla_grad(ck, i, d)
    return mu * (nabla_grad_non_ref + nabla_grad_non_ref.T) + lambda_bar * np.trace(nabla_grad_non_ref) * np.identity(2)


def sigma_bq(e_young: float, nu_poisson: float, ck: np.ndarray, i: int, d: int, phi_jac_inv: np.ndarray,
             x: float, y: float):
    # get Lambert's coeffs.
    mu, lambda_bar = get_lambda_mu(e_young, nu_poisson)
    nabla_grad_non_ref = phi_jac_inv.T @ assembly.quadrilateral.bilinear.nabla_grad(x, y, ck, i, d)
    return mu * (nabla_grad_non_ref + nabla_grad_non_ref.T) + lambda_bar * np.trace(nabla_grad_non_ref) * np.identity(2)


def get_element_stress(solver: BaseSolver, compute_for_rb: bool = False) -> np.ndarray:
    n_el = solver.tri.shape[0]
    element_stress = np.zeros((n_el, 2, 2))
    for el_nr, nk in enumerate(solver.tri):
        # nk : node-numbers for the k'th triangle or quadrilateral
        # the points of the triangle or quadrilateral
        # p1 = p[nk[0], :], uh1 = uh.values[nk[0], :]
        # p2 = p[nk[1], :], uh2 = uh.values[nk[1], :]
        # p3 = p[nk[2], :], uh3 = uh.values[nk[2], :]
        # p4 = p[nk[3], :], uh3 = uh.values[nk[2], :]
        # using indexmap k = 2 * i + d, d=0 for x, 1 for y, i is the node number
        # and basis functions coef. or Jacobin inverse
        if solver.element in ("triangle triangle", "lt"):
            ck = assembly.triangle.linear.get_basis_coef(solver.p[nk, :])
            for i in range(3):
                if compute_for_rb:
                    jac_phi_inv = solver.jac_phi_inv(*solver.p[nk[i], :], *solver.uh_rom.geo_params)
                else:
                    jac_phi_inv = solver.jac_phi_inv(*solver.p[nk[i], :], *solver.uh.geo_params)
                # i gives basis, d gives dimension
                for d in range(2):
                    if compute_for_rb:
                        element_stress[el_nr, :, :] += solver.uh_rom.values[nk[i], d] \
                                                       * sigma_lt(solver.uh_rom.e_young,
                                                                  solver.uh_rom.nu_poisson,
                                                                  ck, i, d, jac_phi_inv)
                    else:
                        element_stress[el_nr, :, :] += solver.uh.values[nk[i], d] \
                                                       * sigma_lt(solver.uh.e_young,
                                                                  solver.uh.nu_poisson,
                                                                  ck, i, d, jac_phi_inv)
        else:
            ck = assembly.quadrilateral.bilinear.get_basis_coef(solver.p[nk, :])
            for i in range(4):
                if compute_for_rb:
                    jac_phi_inv = solver.jac_phi_inv(*solver.p[nk[i], :], *solver.uh_rom.geo_params)
                else:
                    jac_phi_inv = solver.jac_phi_inv(*solver.p[nk[i], :], *solver.uh.geo_params)
                # i gives basis, d gives dimension
                for d in range(2):
                    if compute_for_rb:
                        element_stress[el_nr, :, :] += solver.uh_rom.values[nk[i], d] \
                                                       * sigma_bq(solver.uh_rom.e_young,
                                                                  solver.uh_rom.nu_poisson,
                                                                  ck, i, d, jac_phi_inv, *solver.p[nk[i], :])
                    else:
                        element_stress[el_nr, :, :] += solver.uh.values[nk[i], d] \
                                                       * sigma_bq(solver.uh.e_young,
                                                                  solver.uh.nu_poisson,
                                                                  ck, i, d, jac_phi_inv, *solver.p[nk[i], :])
    return element_stress


def get_node_neighbour_elements(node_nr: int, tri: np.ndarray) -> np.ndarray:
    return np.argwhere(tri == node_nr)[:, 0]


def get_nodal_stress(solver: BaseSolver, compute_for_rb: bool = False):
    n_nodes = solver.p.shape[0]
    # recover the element stress
    element_stress = get_element_stress(solver, compute_for_rb=compute_for_rb)
    nodal_stress = np.zeros((n_nodes, 2, 2))
    for node_nr in np.unique(solver.tri):
        # get index of the neighbour elements
        node_n_el = get_node_neighbour_elements(node_nr, solver.tri)
        # recovery, calculate the mean value.
        nodal_stress[node_nr, :, :] = np.mean(element_stress[node_n_el, :, :], axis=0)
    if compute_for_rb:
        solver.uh_rom.set_nodal_stress(nodal_stress)
    else:
        solver.uh.set_nodal_stress(nodal_stress)


def get_von_mises_stress(solver: BaseSolver, compute_for_rb: bool = False):
    if compute_for_rb:
        n_nodes = solver.uh_rom.nodal_stress.shape[0]
    else:
        n_nodes = solver.uh.nodal_stress.shape[0]
    von_mises = np.zeros(n_nodes)
    for node_nr in range(n_nodes):
        # deviatoric stress
        if compute_for_rb:
            s = solver.uh_rom.nodal_stress[node_nr, :, :] \
                - np.trace(solver.uh_rom.nodal_stress[node_nr, :, :]) * np.identity(2) / 3
        else:
            s = solver.uh.nodal_stress[node_nr, :, :] \
                - np.trace(solver.uh.nodal_stress[node_nr, :, :]) * np.identity(2) / 3
        von_mises[node_nr] = np.sqrt(3 / 2 * np.sum(s * s))
    if compute_for_rb:
        solver.uh_rom.set_von_mises(von_mises)
    else:
        solver.uh.set_von_mises(von_mises)
