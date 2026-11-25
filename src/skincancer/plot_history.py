"""Training curves of a model's seeds: loss, accuracy and validation AUC against epochs.

    python -m skincancer.plot_history --exp skinnet-idea-t          # -> results/<exp>/curves.png

Accuracy is logged from 22 Nov 2026 on (train.py); older runs show loss and validation AUC only.
Training accuracy is measured on the augmented, class-balanced batches the model trains on, so it is
not directly comparable with validation accuracy, which is on the real class mix (about 18% cancer:
always answering "not cancer" scores 82%). Validation AUC is what early stopping uses.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .paths import exp_dir, results_dir as results_dir_of


def plot(exp: str, runs_dir: str = "runs", results_dir: str = "results") -> Path:
    seeds = sorted(exp_dir(exp, runs_dir).glob("s[0-9]*"))
    hists = {s.name: pd.read_csv(s / "history.csv") for s in seeds if (s / "history.csv").exists()}
    if not hists:
        raise SystemExit(f"no history.csv under {exp_dir(exp, runs_dir)}")
    panels = [("Loss", [("loss", "train"), ("val_loss", "validation")]),
              ("Accuracy", [("train_acc", "train"), ("val_acc", "validation")]),
              ("Validation AUC", [("val_auc", "validation")])]
    panels = [(t, [(c, l) for c, l in cols if any(c in h for h in hists.values())]) for t, cols in panels]
    panels = [p for p in panels if p[1]]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.8), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (title, cols) in zip(axes[0], panels):
        for k, (seed, h) in enumerate(hists.items()):
            for col, lab in cols:
                if col in h:
                    ax.plot(h["epoch"] + 1, h[col], color=colors[k % len(colors)],
                            linestyle="-" if lab == "train" else "--", label=f"{seed} {lab}")
            if title == "Validation AUC":
                b = h["val_auc"].idxmax()
                ax.plot(h["epoch"][b] + 1, h["val_auc"][b], "o", color=colors[k % len(colors)])
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"{exp}: training curves (dot: the epoch kept)")
    fig.tight_layout()
    out = results_dir_of(exp, results_dir) / "curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", required=True)
    a = ap.parse_args(argv)
    for e in a.exp:
        print(plot(e))


if __name__ == "__main__":
    main()
