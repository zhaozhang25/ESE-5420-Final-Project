"""Experiment configuration, loaded from a YAML file in configs/."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Config:
    name: str
    seed: int
    data_path: Path
    drop_duplicates: bool
    test_size: float
    n_folds: int
    splits_path: Path
    c_grid: tuple
    max_iter: int
    results_dir: Path
    figure_path: Path


def load_config(path):
    """Read a config. Relative paths resolve against the repo root (the parent of configs/)."""
    path = Path(path).resolve()
    root = path.parent.parent
    raw = yaml.safe_load(path.read_text())
    grid = raw["logreg"]["c_grid"]
    return Config(
        name=raw["name"],
        seed=raw["seed"],
        data_path=root / raw["data"]["path"],
        drop_duplicates=raw["data"]["drop_duplicates"],
        test_size=raw["split"]["test_size"],
        n_folds=raw["split"]["n_folds"],
        splits_path=root / raw["split"]["path"],
        c_grid=tuple(np.logspace(grid["min_exp"], grid["max_exp"], grid["num"])),
        max_iter=raw["logreg"]["max_iter"],
        results_dir=root / raw["output"]["results_dir"],
        figure_path=root / raw["output"]["figure"],
    )
