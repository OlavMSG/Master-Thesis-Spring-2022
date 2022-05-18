# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tqdm
from matrix_lsq import DiskStorage, Snapshot

from .base_solver import BaseSolver


class RBModelSaver:
    storage: DiskStorage
    root: Path

    def __init__(self, root: Path):
        self.root = root
        self.storage = DiskStorage(root)

        assert len(self.storage) == 0

    def __call__(self, solver: BaseSolver, pod_v: np.ndarray):
        v_root = self.root / "pod_v"
        v_root.mkdir(parents=True, exist_ok=True)
        Snapshot(v_root, pod_v=pod_v)
        if solver.has_non_homo_dirichlet:
            for a1_rom, a2_rom, f0_rom, f1_dir_rom, f2_dir_rom in tqdm.tqdm(zip(solver.a1_rom_list,
                                                                                solver.a2_rom_list,
                                                                                solver.f0_rom_list,
                                                                                solver.f1_dir_rom_list,
                                                                                solver.f2_dir_rom_list),
                                                                            desc="Saving RB model"):
                self.storage.append(a1_rom=a1_rom, a2_rom=a2_rom, f0_rom=f0_rom,
                                    f1_dir_rom=f1_dir_rom, f2_dir_rom=f2_dir_rom)
        else:
            for a1_rom, a2_rom, f0_rom in tqdm.tqdm(zip(solver.a1_rom_list,
                                                        solver.a2_rom_list,
                                                        solver.f0_rom_list),
                                                    desc="Saving RB model"):
                self.storage.append(a1_rom=a1_rom, a2_rom=a2_rom, f0_rom=f0_rom)


class RBModelLoader:
    storage: DiskStorage
    root: Path

    def __init__(self, root: Path):
        self.root = root
        self.storage = DiskStorage(root)

        assert len(self.storage) != 0

    def __call__(self, solver: BaseSolver) -> np.ndarray:
        solver.a1_rom_list = [0] * len(self.storage)
        solver.a2_rom_list = [0] * len(self.storage)
        solver.f0_rom_list = [0] * len(self.storage)
        if solver.has_non_homo_dirichlet:
            solver.f1_dir_rom_list = [0] * len(self.storage)
            solver.f2_dir_rom_list = [0] * len(self.storage)
        for i, snapshot in tqdm.tqdm(enumerate(self.storage), desc="Loading RB model"):
            solver.a1_rom_list[i] = snapshot["a1_rom"]
            solver.a2_rom_list[i] = snapshot["a2_rom"]
            solver.f0_rom_list[i] = snapshot["f0_rom"]
            if solver.has_non_homo_dirichlet:
                solver.f1_dir_rom_list[i] = snapshot["f1_dir_rom"]
                solver.f2_dir_rom_list[i] = snapshot["f2_dir_rom"]
        v_root = self.root / "pod_v"
        return Snapshot(v_root)["pod_v"]
