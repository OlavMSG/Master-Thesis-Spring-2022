# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
based on Specialization-Project-fall-2021
"""

# Default tolerance for checking
default_tol = 1e-14
# Default plate limits
# we will use the plate [self._plate_limits[0], self._plate_limits[1]]^2
plate_limits = (0, 1)
# Default e_young-nu_poisson material_grid
e_nu_grid = 5
# default pod sample_mode
pod_sampling_mode = "Uniform"
implemented_pod_modes = ("Uniform", "Gauss-Lobatto")
# Ranges for parameters
e_young_range = (10e3, 310e3)  # MPa
e_young_unit = "MPa"
nu_poisson_range = (0, 0.4)
nu_poisson_unit = "1"
# epsilon for pod
eps_pod = 1e-2
# max cut value for n_rom
n_rom_cut = "rank"  # = 1e-12
# default names for saved files
"""file_names_dict = {"a1": "a1_mat.npz",
                   "a2": "a2_mat.npz",
                   "f_load_lv": "f_load_lv.npy",
                   "f_load_neumann": "f_load_neumann.npy",
                   "dirichlet_edge": "dirichlet_edge.npy",
                   "p_tri_edge": "p_tri_edge.npz",
                   "rg": "rg_lifting_func.npy",
                   "a1_rom": "a1_rom_mat.npy",
                   "a2_rom": "a2_rom_mat.npy",
                   "f1_dir_rom": "f1_dir_rom.npy",
                   "f2_dir_rom": "f2_dir_rom.npy",
                   "f_load_lv_rom": "f_load_lv_rom.npy",
                   "f_load_neumann_rom": "f_load_neumann_rom.npy",
                   "v": "v_mat_n_max.npy",
                   "sigma2": "singular_values_vec.npy",
                   "pod_parameters": "pod_parameters.npy"}"""
