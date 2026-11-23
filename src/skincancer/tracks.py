"""Where a model's files live. The two tracks keep separate folders (DECISIONS.md, section 28):

    runs/<track>/<name>/s<seed>/      runs/skinnet/skinnet-c5/s0, runs/panderm/panderm-c2/avg
    results/<track>/<name>/           metrics.csv, curves.png
    checkpoints/<track>/<name>/       kept weights (local)

The track comes from the model's name: names starting with `panderm` are PanDerm's, every other
name is SkinNet's (the from-scratch track, and the default).
"""
from __future__ import annotations

from pathlib import Path

TRACKS = ("skinnet", "panderm")


def track(name: str) -> str:
    return "panderm" if name.startswith("panderm") else "skinnet"


def exp_dir(name: str, runs_dir: str | Path = "runs") -> Path:
    """runs/<track>/<name>: the model's seeds (s0, s1, ...) and their average (avg)."""
    return Path(runs_dir) / track(name) / name


def results_dir(name: str, results_root: str | Path = "results") -> Path:
    return Path(results_root) / track(name) / name
