"""Where a model's files live, by its name:

    runs/<name>/s<seed>/      one seed: best.pt, config.yaml, history.csv, summary.json, preds/
    runs/<name>/avg/          the seed average's predictions (derive ensemble)
    results/<name>/           metrics.csv, curves.png
    checkpoints/<name>/       a kept model's weights (local)
"""
from __future__ import annotations

from pathlib import Path


def exp_dir(name: str, runs_dir: str | Path = "runs") -> Path:
    """runs/<name>: the model's seeds (s0, s1, ...) and their average (avg)."""
    return Path(runs_dir) / name


def results_dir(name: str, results_root: str | Path = "results") -> Path:
    return Path(results_root) / name
