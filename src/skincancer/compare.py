"""Paired comparison of two models on the same images.

    python -m skincancer.compare --a skinnet-idea-t --b skinnet-lodo-pad --splits test --datasets pad_ufes_20
    python -m skincancer.compare --a skinnet-c2 --b skinnet-c3 --splits test ddi_test midas_test
    # one kind of cancer against every non-cancer in the set (here mycosis fungoides, section 29)
    python -m skincancer.compare --a skinnet-c5 --b skinnet-idea-v --splits ddi_hold --positives "mycosis.fungoides"

Two separate confidence intervals overlap long before a difference stops being real, because
most of their width is which images happened to be in the test set, and both models share
those images. Here every bootstrap resample of groups is scored by both models and the
difference (b minus a, each averaged over its seeds) gets its own 95% interval. An interval
that excludes zero is a difference the data supports.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import runs
from .evaluate import load_preds, run_config
from .labels import CLASSES
from .paths import exp_dir


def _scores(exp_dir: Path, split: str, ids=None):
    cfg = run_config(exp_dir)
    classes = CLASSES[cfg["data"]["label_col"]]
    _, frames = load_preds(exp_dir, split)
    if not frames:
        raise FileNotFoundError(f"{exp_dir}: no predictions for {split}")
    ids = frames[0]["image_id"] if ids is None else ids
    cols = [f"p_{c}" for c in classes]
    return ids, [M.p_malignant(f.set_index("image_id").loc[ids, cols].to_numpy(), classes) for f in frames]


def compare(a: str, b: str, split: str, manifest: pd.DataFrame, runs_dir="runs", n_boot: int = 2000, seed: int = 0,
            datasets=None, positives: str | None = None) -> list[dict]:
    """datasets: score only the images from these sources (e.g. the one a leave-one-dataset-out model left out).
    positives: keep only the cancers whose diagnosis matches this regular expression (case ignored);
    every non-cancer stays."""
    ids, sa = _scores(exp_dir(a, runs_dir), split)
    _, sb = _scores(exp_dir(b, runs_dir), split, ids)
    meta = manifest.set_index("image_id").loc[ids]
    y = meta["label_binary"].astype(float).to_numpy()
    keep = ~np.isnan(y)
    if datasets:
        keep &= meta["dataset"].isin(datasets).to_numpy()
    if positives:
        hit = meta["diagnosis"].astype(str).str.contains(positives, case=False, regex=True).to_numpy()
        keep &= (y != 1) | hit
    y, src = y[keep], meta["dataset"].to_numpy()[keep]
    sa, sb = [s[keep] for s in sa], [s[keep] for s in sb]
    groups = meta["group_id"].astype(str).to_numpy()[keep]
    fns = {"auc": lambda yy, s, sr: M.auc(yy == 1, s)}
    if len(np.unique(src)) > 1:
        fns["source_auc"] = M.source_auc
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    rows = []
    for name, f in fns.items():
        def both(idx):
            va = np.mean([f(y[idx], s[idx], src[idx]) for s in sa])
            vb = np.mean([f(y[idx], s[idx], src[idx]) for s in sb])
            return va, vb
        pa, pb = both(np.arange(len(y)))
        diffs = []
        for _ in range(n_boot):
            idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
            va, vb = both(idx)
            if not (np.isnan(va) or np.isnan(vb)):
                diffs.append(vb - va)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"split": split, "metric": name, "a": pa, "b": pb, "diff": pb - pa, "ci_lo": lo, "ci_hi": hi,
                     "supported": "yes" if lo > 0 or hi < 0 else "no", "n": int(len(y))})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="reference experiment")
    ap.add_argument("--b", required=True, nargs="+", help="experiment(s) compared with it")
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--datasets", nargs="+", help="only the images from these sources")
    ap.add_argument("--positives", help="only the cancers whose diagnosis matches this regex; all non-cancers stay")
    args = ap.parse_args(argv)
    cfg = run_config(exp_dir(args.a, args.runs_dir))
    manifest = runs.read_manifest(cfg["manifest"])
    rows = []
    for b in args.b:
        for split in args.splits:
            for r in compare(args.a, b, split, manifest, args.runs_dir, args.n_boot, datasets=args.datasets,
                             positives=args.positives):
                rows.append({"b": b, **{k: v for k, v in r.items() if k != "b"}, "b_value": r["b"]})
    out = pd.DataFrame(rows).rename(columns={"a": "a_value"})
    out = out[["b", "split", "metric", "a_value", "b_value", "diff", "ci_lo", "ci_hi", "supported", "n"]]
    print(f"reference: {args.a}")
    print(out.round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    main()
