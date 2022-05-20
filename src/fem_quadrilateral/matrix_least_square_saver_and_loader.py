# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import tqdm
from matrix_lsq import DiskStorage, Snapshot

from .base_solver import BaseSolver


class MLSSaver:
    storage: DiskStorage
    root: Path

    def __init__(self, root: Path):
        self.root = root
        self.storage = DiskStorage(root)

        assert len(self.storage) == 0

    def __call__(self, solver: BaseSolver):
        if solver.has_non_homo_dirichlet:
            for a1, a2, f0, f1_dir, f2_dir in tqdm.tqdm(zip(solver.mls.a1_list,
                                                            solver.mls.a2_list,
                                                            solver.mls.f0_list,
                                                            solver.mls.f1_dir_list,
                                                            solver.mls.f2_dir_list),
                                                        desc="Saving MLS matrices and vectors"):
                self.storage.append(a1=a1, a2=a2, f0=f0, f1_dir=f1_dir, f2_dir=f2_dir)
        else:
            for a1, a2, f0 in tqdm.tqdm(zip(solver.mls.a1_list,
                                            solver.mls.a2_list,
                                            solver.mls.f0_list,
                                            ),
                                        desc="Saving MLS matrices and vectors"):
                self.storage.append(a1=a1, a2=a2, f0=f0)


class MLSLoader:
    storage: DiskStorage
    root: Path

    def __init__(self, root: Path):
        self.root = root
        self.storage = DiskStorage(root)

        assert len(self.storage) != 0

    def __call__(self, solver: BaseSolver):
        solver.mls.a1_list = [0] * len(self.storage)
        solver.mls.a2_list = [0] * len(self.storage)
        solver.mls.f0_list = [0] * len(self.storage)
        if solver.has_non_homo_dirichlet:
            solver.mls.f1_dir_list = [0] * len(self.storage)
            solver.mls.f2_dir_list = [0] * len(self.storage)
        for i, snapshot in tqdm.tqdm(enumerate(self.storage), desc="Loading MLS matrices and vectors"):
            solver.mls.a1_list[i] = snapshot["a1"]
            solver.mls.a2_list[i] = snapshot["a2"]
            solver.mls.f0_list[i] = snapshot["f0"]
            if solver.has_non_homo_dirichlet:
                solver.mls.f1_dir_list[i] = snapshot["f1_dir_rom"]
                solver.mls.f2_dir_list[i] = snapshot["f2_dir_rom"]
        solver.mls.num_kept = len(solver.mls.a1_list)
