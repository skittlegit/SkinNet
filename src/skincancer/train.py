"""Train SkinNet from random weights: every layer, one learning rate, no pretrained weights.

    python -m skincancer.train --config configs/skinnet/skinnet.yaml --seed 0

Writes runs/<track>/<name>/s<seed>/: best.pt, config.yaml, history.csv, summary.json and
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


def make_scheduler(opt, epochs: int, warmup: int = 0, head_epochs: int = 0, head_lr: float | None = None):
    """Stepped once per epoch: linear warm-up over `warmup` epochs, then cosine decay to zero.
    With head_epochs (PanDerm track, DECISIONS.md section 24) the first head_epochs train only
    the parameter groups marked `head`, at the constant rate head_lr; every other group's rate
    is zero until then, and the warm-up and cosine run over the epochs that remain."""
    rest = max(epochs - head_epochs - warmup, 1)

    def body(e):
        e -= head_epochs
        if e < 0:
            return 0.0
        return (e + 1) / warmup if e < warmup else 0.5 * (1 + math.cos(math.pi * (e - warmup) / rest))

    def head(g):
        base = g["lr"]   # read now: the scheduler changes g["lr"] as it steps
        return lambda e: head_lr / base if e < head_epochs else body(e)

    return torch.optim.lr_scheduler.LambdaLR(
        opt, [head(g) if g.get("head") and head_epochs else body for g in opt.param_groups])


class WeightAverage:
    """A running average of the network's weights (idea PF, DECISIONS.md section 24): every
    `every` training steps avg <- avg + (1 - decay**every) * (weights - avg), so `decay` is the
    per-step rate and about 1 / (1 - decay) steps carry most of the weight. Kept in CPU memory: a
    second copy of PanDerm Large on the GPU would not fit in 8 GB. swap() exchanges the network's
    weights with the average, in place; a second swap() puts them back."""

    def __init__(self, model, decay: float, every: int = 1):
        self.params = list(model.parameters())
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


def param_groups(model, lr: float, layer_decay: float | None = None) -> list[dict]:
    """Parameter groups for AdamW. The linear head is its own group, marked `head`. With
    layer_decay (PanDerm's ViT only) each transformer block's rate is the one above it times
    layer_decay: the final norm trains at lr, block 11 at lr * decay, ..., the patch embedding
    and class token at lr * decay^13, so the early, general features move least."""
    groups = [{"params": list(model.head.parameters()), "lr": lr, "head": True}]
    if not layer_decay:
        return groups + [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": lr}]
    from .models import PanDermBase
    if not isinstance(model.backbone, PanDermBase):
        raise ValueError("train.layer_decay is set up for PanDerm's ViTs only")
    depth = len(model.backbone.net.blocks)
    by_layer: dict[int, list] = {}
    for name, p in model.backbone.net.named_parameters():
        if not p.requires_grad:
            continue
        layer = int(name.split(".")[1]) + 1 if name.startswith("blocks.") else depth + 1 if name.startswith("norm.") else 0
        by_layer.setdefault(layer, []).append(p)
    return groups + [{"params": ps, "lr": lr * layer_decay ** (depth + 1 - layer)} for layer, ps in sorted(by_layer.items())]


def train(cfg: dict) -> dict:
    runs.set_seed(cfg["seed"])
    rdir = runs.run_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    runs.save_config(cfg, rdir / "config.yaml")
    dev = runs.device()
    classes = runs.classes_of(cfg)
    t = cfg["train"]
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
    if cfg["model"].get("init"):   # idea L's start from self-supervised weights, removed 23 Nov 2026 (section 28)
        raise ValueError("model.init is no longer supported: idea L's pretraining was removed (DECISIONS.md, section 28)")
    model = model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{cfg['name']} s{cfg['seed']}] train={len(train_df)} val={len(val_df)} "
          f"params={n_params / 1e6:.2f}M device={dev}")
    tr_loader = runs.make_loader(train_df, cfg, train=True, seed=cfg["seed"], drop_last=True,
                                 sample_weights=sampling)
    val_loader = runs.make_loader(val_df, cfg, train=False)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=dev),
                                    label_smoothing=float(t["label_smoothing"]))
    opt = torch.optim.AdamW(param_groups(model, t["lr"], t.get("layer_decay")), lr=t["lr"],
                            weight_decay=t["weight_decay"])
    sched = make_scheduler(opt, t["epochs"], t["warmup_epochs"], t.get("head_epochs", 0), t.get("head_lr"))
    use_amp = t["amp"] and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = WeightAverage(model, t["ema_decay"], t.get("ema_every", 1)) if t.get("ema_decay") else None

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
            with torch.autocast(dev.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            hits += (out.argmax(1) == y).sum().item()   # logged only: accuracy on the (augmented, resampled) batches
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update()
            tot += loss.item() * len(y)
            n += len(y)
        lr = opt.param_groups[-1]["lr"]   # the top of the backbone (the head's is the same after head_epochs)
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
    ap.add_argument("--config", default="configs/skinnet/skinnet.yaml")
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
