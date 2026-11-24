"""Where a model's files live:

    runs/skinnet/<name>/s<seed>/      runs/skinnet/skinnet-c5/s0, runs/skinnet/skinnet-c5/avg
    results/skinnet/<name>/           metrics.csv, curves.png
    checkpoints/skinnet/<name>/       kept weights (local)

The `skinnet/` level is kept from when both tracks shared one repository (DECISIONS.md, sections
28 and 31), so every path in the test-set ledger and the docs still holds.
"""
from __future__ import annotations

from pathlib import Path

TRACK = "skinnet"


def exp_dir(name: str, runs_dir: str | Path = "runs") -> Path:
    """runs/skinnet/<name>: the model's seeds (s0, s1, ...) and their average (avg)."""
    return Path(runs_dir) / TRACK / name


def results_dir(name: str, results_root: str | Path = "results") -> Path:
    return Path(results_root) / TRACK / name
