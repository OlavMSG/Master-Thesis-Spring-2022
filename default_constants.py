# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
based on Specialization-Project-fall-2021
"""

# default geometry ranges
QS_range = (-0.1, 0.1)  # Must be abs(...)< 1/6
DR_range = (-0.3, 0.3)  # Must be abs(...)< 1/2
SR_range = (0.1, 5.1)   # Must be > 0

# Default tolerance for checking
default_tol = 1e-14
# Default plate limits
# we will use the plate [plate_limits[0], plate_limits[1]]^2
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

