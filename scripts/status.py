"""What the training is doing: every seed that has started but not finished (it has a last.pt),
how far it is, its best validation score so far and roughly how long it has left, then the
last lines of runs/queue.log. Read-only.

    python scripts/status.py           # usually through: bash scripts/queue.sh status
    python scripts/status.py stopped   # the queue is known not to run: no seed is "training"
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


def fmt(secs: float) -> str:
    h, m = divmod(int(round(secs / 60)), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def seed_progress(sdir: Path) -> dict:
    """One unfinished seed: epochs done of the planned number, mean time per epoch, best score."""
    cfg = yaml.safe_load((sdir / "config.yaml").read_text(encoding="utf-8"))
    hist = pd.read_csv(sdir / "history.csv") if (sdir / "history.csv").exists() else pd.DataFrame()
    planned = int(cfg["train"]["epochs"])
    done = len(hist)
    secs = float(hist["secs"].mean()) if done else float("nan")
    best = hist.loc[hist["val_auc"].idxmax()] if done else None
    return {"exp": sdir.parent.name, "seed": sdir.name, "done": done, "planned": planned, "secs": secs,
            "left": (planned - done) * secs if done else float("nan"),
            "best": None if best is None else (int(best["epoch"]), float(best["val_auc"])),
            "patience": int(cfg["train"].get("patience") or 0),
            "updated": (sdir / "last.pt").stat().st_mtime}


def seeds_left(exp_dir: Path, seeds=(0, 1, 2)) -> list[str]:
    """Seeds of the same model not started yet (no folder or no history)."""
    return [f"s{s}" for s in seeds if not (exp_dir / f"s{s}" / "summary.json").exists()
            and not (exp_dir / f"s{s}" / "last.pt").exists()]


def report(runs_dir: Path = Path("runs"), log_lines: int = 8, queue_running: bool = True) -> str:
    out = []
    for last in sorted(runs_dir.glob("*/s*/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        p = seed_progress(last.parent)
        age = time.time() - p["updated"]
        recent = p["secs"] == p["secs"] and age < 3 * p["secs"] + 600
        state = "training" if queue_running and recent else "stopped"
        line = f"{p['exp']} {p['seed']}: {state}, epoch {p['done']} of {p['planned']}"
        if p["done"]:
            line += f", {p['secs']:.0f} s per epoch, best val {p['best'][1]:.4f} at epoch {p['best'][0]}"
            line += f"; at most {fmt(p['left'])} left" + (" (it stops early if val stops improving)" if p["patience"] else "")
        rest = seeds_left(last.parent.parent)
        if rest and p["done"]:
            epochs_run = [json.loads((s / "summary.json").read_text())["epochs_run"]
                          for s in last.parent.parent.glob("s*") if (s / "summary.json").exists()]
            per_seed = (sum(epochs_run) / len(epochs_run) if epochs_run else p["planned"]) * p["secs"]
            line += f"; then {', '.join(rest)} (about {fmt(per_seed * len(rest))})"
        out.append(line)
    if not out:
        out.append("no seed is in progress")
    log = runs_dir / "queue.log"
    if log.exists():
        tail = [l.rstrip() for l in log.read_text(encoding="utf-8", errors="replace").splitlines()
                if l.startswith("[")][-log_lines:]
        out += ["", f"last lines of {log.as_posix()}:"] + ["  " + l for l in tail]
    return "\n".join(out)


if __name__ == "__main__":
    print(report(queue_running="stopped" not in sys.argv[1:]))
