"""Train SkinNet from random weights: every layer, one learning rate, no pretrained weights.

    python -m skincancer.train --config configs/skinnet.yaml --seed 0

Writes runs/<name>/s<seed>/: best.pt, config.yaml, history.csv, summary.json and
preds/<split>.csv for every split in eval.splits.
"""
from __future__ import annotations

import argparse
import gc
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import runs
from .regularize import mix_batch, sam_step, soft_cross_entropy
from .config import load_config
from .metrics import auc, source_auc


def class_weights(counts: np.ndarray, weighting: str) -> np.ndarray:
    """Loss weight per class. balanced: every class counts equally in total, whatever its size."""
    if weighting == "none":
        return np.ones(len(counts))
    if weighting != "balanced":
        raise ValueError(f"unknown train.class_weighting {weighting!r}")
    pi = counts / counts.sum()
    w = (1.0 / len(counts)) / np.maximum(pi, 1e-12)
    return w / (w * pi).sum()


def sample_weights(df: pd.DataFrame, mode: str, label_col: str,
                   dataset_weight: dict | None = None) -> np.ndarray | None:
    """Per-image sampling weights for training (mean 1), or None for one plain pass per epoch.

    source_class: inside each dataset, cancer and not cancer are drawn equally often, so the
    kind of photo no longer predicts the class. A dataset keeps its share of the draws, times
    `dataset_weight[dataset]` if given (e.g. {milk10k: 2} draws MILK10k twice as often)."""
    if mode == "none":
        return None
    if mode != "source_class":
        raise ValueError(f"unknown train.sampling {mode!r}")
    w = np.zeros(len(df))
    for _, cell in df.groupby(["dataset", label_col]):
        in_dataset = df["dataset"] == cell["dataset"].iloc[0]
        w[cell.index] = in_dataset.sum() / df.loc[in_dataset, label_col].nunique() / len(cell)
    for name, factor in (dataset_weight or {}).items():
        if not (df["dataset"] == name).any():
            raise ValueError(f"train.dataset_weight names a dataset not in training: {name!r}")
        w[(df["dataset"] == name).to_numpy()] *= factor
    return w / w.mean()


def selection_score(y: np.ndarray, p_cancer: np.ndarray, source: np.ndarray, metric: str) -> float:
    """The validation number that picks the checkpoint: plain AUC, or the within-source AUC."""
    if metric == "auc":
        return auc(y == 1, p_cancer)
    if metric == "source_auc":
        return source_auc(y.astype(float), p_cancer, source)
    raise ValueError(f"unknown train.select_metric {metric!r}")


def make_scheduler(opt, epochs: int, warmup: int = 0):
    """Stepped once per epoch: linear warm-up over `warmup` epochs, then cosine decay to zero."""
    rest = max(epochs - warmup, 1)

    def factor(e):
        return (e + 1) / warmup if e < warmup else 0.5 * (1 + math.cos(math.pi * (e - warmup) / rest))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


class WeightAverage:
    """A running average of the network's weights (idea PF, DECISIONS.md section 24): every
    `every` training steps avg <- avg + (1 - decay**every) * (weights - avg), so `decay` is the
    per-step rate and about 1 / (1 - decay) steps carry most of the weight. Kept in CPU memory: a
    second copy of PanDerm Large on the GPU would not fit in 8 GB. swap() exchanges the network's
    weights with the average, in place; a second swap() puts them back. Batch norm's running
    statistics (SkinNet, idea W) are averaged with the weights, so the averaged weights are never
    paired with statistics gathered for other weights."""

    def __init__(self, model, decay: float, every: int = 1):
        self.params = list(model.parameters()) + [b for b in model.buffers() if b.dtype.is_floating_point]
        self.avg = [p.detach().to("cpu", copy=True) for p in self.params]
        pin = self.params[0].is_cuda
        self.buf = [torch.empty_like(a, pin_memory=pin) for a in self.avg]
        self.weight, self.every, self.steps = 1 - decay ** every, every, 0

    @torch.no_grad()
    def update(self):
        self.steps += 1
        if self.steps % self.every:
            return
        for b, p in zip(self.buf, self.params):
            b.copy_(p, non_blocking=True)
        if self.params[0].is_cuda:
            torch.cuda.synchronize()
        torch._foreach_lerp_(self.avg, self.buf, self.weight)

    @torch.no_grad()
    def swap(self):
        for b, a, p in zip(self.buf, self.avg, self.params):
            b.copy_(p)
            p.copy_(a)
            a.copy_(b)

    def state_dict(self) -> dict:
        return {"avg": self.avg, "steps": self.steps}

    def load_state_dict(self, st: dict):
        for a, s in zip(self.avg, st["avg"]):
            a.copy_(s)
        self.steps = st["steps"]


def train(cfg: dict) -> dict:
    t = cfg["train"]
    for key in ("layer_decay", "head_epochs"):   # PanDerm's options, in skittlegit/PanDerm since section 31
        if t.get(key):
            raise ValueError(f"train.{key} is for PanDerm's ViTs (skittlegit/PanDerm)")
    if cfg["model"].get("init"):   # idea L's start from self-supervised weights, removed 23 Nov 2026 (section 28)
        raise ValueError("model.init is no longer supported: idea L's pretraining was removed (DECISIONS.md, section 28)")
    runs.set_seed(cfg["seed"])
    rdir = runs.run_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    runs.save_config(cfg, rdir / "config.yaml")
    dev = runs.device()
    classes = runs.classes_of(cfg)
    manifest = pd.read_csv(cfg["manifest"])
    train_df = runs.labelled(runs.split_rows(manifest, cfg["data"]["train_splits"]), cfg)
    if cfg["data"].get("extra_train"):   # training-only data kept outside the manifest (data/extra.py)
        from .data.extra import load_extra
        extra = load_extra(cfg["data"]["extra_train"], cfg["data"]["label_col"], runs.classes_of(cfg))
        train_df = pd.concat([train_df, extra[train_df.columns.intersection(extra.columns)]], ignore_index=True)
    val_df = runs.labelled(runs.split_rows(manifest, cfg["data"]["val_splits"]), cfg)
    excl = cfg["data"].get("exclude_datasets") or []   # leave-one-dataset-out: kept out of training and validation
    if excl:
        train_df = train_df[~train_df["dataset"].isin(excl)]
        val_df = val_df[~val_df["dataset"].isin(excl)]
        print(f"[{cfg['name']} s{cfg['seed']}] datasets left out: {excl}")
    label_col = cfg["data"]["label_col"]
    counts = train_df[label_col].value_counts().reindex(classes, fill_value=0).to_numpy(float)
    if (counts == 0).any():
        raise ValueError(f"empty class in training data: {dict(zip(classes, counts))}")
    weights = class_weights(counts, t["class_weighting"])
    train_df = train_df.reset_index(drop=True)
    sampling = sample_weights(train_df, t.get("sampling", "none"), label_col, t.get("dataset_weight"))
    if sampling is not None and t["class_weighting"] != "none":
        raise ValueError("train.sampling already balances the classes: set train.class_weighting to none")
    select_metric = t.get("select_metric", "auc")
    val_source = val_df["dataset"].to_numpy()

    model = runs.build_model(cfg)
    model = model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{cfg['name']} s{cfg['seed']}] train={len(train_df)} val={len(val_df)} "
          f"params={n_params / 1e6:.2f}M device={dev}")
    tr_loader = runs.make_loader(train_df, cfg, train=True, seed=cfg["seed"], drop_last=True,
                                 sample_weights=sampling)
    val_loader = runs.make_loader(val_df, cfg, train=False)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=dev),
                                    label_smoothing=float(t["label_smoothing"]))
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = make_scheduler(opt, t["epochs"], t["warmup_epochs"])
    use_amp = t["amp"] and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = WeightAverage(model, t["ema_decay"], t.get("ema_every", 1)) if t.get("ema_decay") else None
    mix = t.get("mix") or {}                 # {mixup: alpha, cutmix: alpha, prob: p} (section 32)
    if mix and t["class_weighting"] != "none":
        raise ValueError("train.mix uses soft targets without class weights: set train.class_weighting to none")
    sam_rho = float(t.get("sam_rho") or 0)   # sharpness-aware minimisation (section 32)
    smoothing = float(t["label_smoothing"])

    def autocast():
        return torch.autocast(dev.type, enabled=use_amp)

    history, best, stale, start = [], {"val_auc": -math.inf}, 0, 0
    resume = rdir / "last.pt"   # written after every epoch, so a stopped run continues where it was
    if resume.exists():
        st = torch.load(resume, map_location="cpu", weights_only=False)   # RNG states must stay on the CPU
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        tr_loader.generator.set_state(st["loader_rng"])
        torch.set_rng_state(st["torch_rng"])
        if torch.cuda.is_available() and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        np.random.set_state(st["np_rng"])
        random.setstate(st["py_rng"])
        if ema is not None:
            ema.load_state_dict(st["ema"])
        history, best, stale, start = st["history"], st["best"], st["stale"], st["epoch"] + 1
        print(f"[{cfg['name']} s{cfg['seed']}] resumed after epoch {st['epoch']}", flush=True)
    for ep in range(start, t["epochs"]):
        if t["patience"] and stale >= t["patience"]:
            break
        t0 = time.time()
        model.train()
        tot, n, hits = 0.0, 0, 0
        for x, y in tr_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if mix:
                x, target = mix_batch(x, y, len(classes), mix.get("mixup", 0), mix.get("cutmix", 0), mix.get("prob", 1.0))

                def loss_of(m, x=x, target=target):
                    o = m(x)
                    return o, soft_cross_entropy(o, target, smoothing)
            else:
                def loss_of(m, x=x, y=y):
                    o = m(x)
                    return o, criterion(o, y)
            opt.zero_grad(set_to_none=True)
            if sam_rho:
                out, loss = sam_step(model, loss_of, opt, scaler, sam_rho, autocast)
            else:
                with autocast():
                    out, loss = loss_of(model)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            hits += (out.argmax(1) == y).sum().item()   # logged only: accuracy on the (augmented, resampled) batches
            if ema is not None:
                ema.update()
            tot += loss.item() * len(y)
            n += len(y)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        model.eval()
        if ema is not None:   # validate, select and keep the averaged weights
            ema.swap()
        pv = runs.infer(model, val_loader)
        yv = np.asarray(val_loader.dataset.labels)
        val_auc = selection_score(yv, pv[:, 1], val_source, select_metric)
        val_loss = float(-np.mean(np.log(np.clip(pv[np.arange(len(yv)), yv], 1e-7, 1.0))))
        rec = {"epoch": ep, "lr": lr, "loss": tot / max(n, 1), "train_acc": hits / max(n, 1), "val_loss": val_loss,
               "val_acc": float((pv.argmax(1) == yv).mean()), "val_auc": val_auc, "secs": round(time.time() - t0, 1)}
        history.append(rec)
        print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()),
              flush=True)
        pd.DataFrame(history).to_csv(rdir / "history.csv", index=False)
        score = -math.inf if np.isnan(val_auc) else val_auc
        if score > best["val_auc"]:
            best = dict(rec, val_auc=score)
            torch.save(model.state_dict(), rdir / "best.pt")
            stale = 0
        else:
            stale += 1
        if ema is not None:
            ema.swap()        # back to the trained weights, which training continues from
        tmp = rdir / "last.pt.tmp"
        torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "loader_rng": tr_loader.generator.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "np_rng": np.random.get_state(), "py_rng": random.getstate(),
                    "history": history, "best": best, "stale": stale,
                    "ema": ema.state_dict() if ema is not None else None}, tmp)
        tmp.replace(resume)
        if t["patience"] and stale >= t["patience"]:
            break
    if not (rdir / "best.pt").exists():
        torch.save(model.state_dict(), rdir / "best.pt")
    runs.write_json({"best": best, "select_metric": select_metric, "epochs_run": len(history), "n_train": len(train_df), "n_val": len(val_df),
                     "n_params": n_params, "class_counts": dict(zip(classes, counts.tolist())),
                     "loss_weights": dict(zip(classes, weights.tolist()))}, rdir / "summary.json")
    resume.unlink(missing_ok=True)   # finished: the resume state is no longer needed (several GB for PanDerm)

    # Shut down the training loaders' worker processes before predicting (15 GB machine).
    del tr_loader, val_loader
    gc.collect()
    model.load_state_dict(torch.load(rdir / "best.pt", map_location=dev))
    model.eval()
    (rdir / "preds").mkdir(exist_ok=True)
    for split in cfg["eval"]["splits"]:
        df = runs.split_rows(manifest, split)
        if len(df) == 0:
            continue
        loader = runs.make_loader(df, cfg, train=False, persistent=False)
        P = runs.infer(model, loader, tta=cfg["eval"].get("tta", False))
        del loader
        runs.prediction_frame(df, P, classes, cfg).to_csv(rdir / "preds" / f"{split}.csv", index=False)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/skinnet.yaml")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    overrides = list(args.set) + ([f"seed={args.seed}"] if args.seed is not None else [])
    cfg = load_config(args.config, overrides)
    if (runs.run_dir(cfg) / "summary.json").exists() and not args.overwrite:
        print(f"{runs.run_dir(cfg)} already finished; pass --overwrite to retrain")
        return
    train(cfg)


if __name__ == "__main__":
    main()
