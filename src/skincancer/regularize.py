"""Training-time regularisers for SkinNet's from-scratch training (DECISIONS.md, section 32).

- mix_batch: mixup or CutMix on a batch, with soft targets (Zhang et al. 2018; Yun et al. 2019).
- soft_cross_entropy: cross-entropy against soft targets, with label smoothing.
- MixStyle: mixes the per-image channel statistics of early feature maps (Zhou et al., ICLR 2021),
  so the network cannot rely on a photo's colour and contrast style, which differs by hospital.
- sam_step: one sharpness-aware minimisation step (Foret et al., ICLR 2021): the gradient is
  taken at the worst nearby weights, which favours flat minima that hold up on new hospitals.

Every random draw uses torch's generator, so seeded runs repeat exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(y: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(y, n).float()


def mix_batch(x: torch.Tensor, y: torch.Tensor, n_classes: int, mixup: float = 0.0, cutmix: float = 0.0,
              prob: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """With probability `prob`, mixes the batch with a shuffled copy of itself: mixup (alpha
    `mixup`) or CutMix (alpha `cutmix`), half the time each when both are on. Returns the
    images and soft targets (rows sum to 1). With nothing applied the targets are one-hot."""
    target = one_hot(y, n_classes)
    use = [m for m, a in (("mixup", mixup), ("cutmix", cutmix)) if a > 0]
    if not use or torch.rand(1).item() >= prob:
        return x, target
    kind = use[0] if len(use) == 1 else use[int(torch.rand(1).item() < 0.5)]
    alpha = mixup if kind == "mixup" else cutmix
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)
    if kind == "mixup":
        x = lam * x + (1 - lam) * x[perm]
    else:
        h, w = x.shape[2:]
        ch, cw = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
        cy, cx = int(torch.randint(h, (1,))), int(torch.randint(w, (1,)))
        y0, y1, x0, x1 = max(cy - ch // 2, 0), min(cy + ch // 2, h), max(cx - cw // 2, 0), min(cx + cw // 2, w)
        x = x.clone()
        x[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)        # the area actually pasted
    return x, lam * target + (1 - lam) * target[perm]


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    """Mean cross-entropy against soft targets; with one-hot targets this equals
    nn.CrossEntropyLoss(label_smoothing=smoothing)."""
    k = logits.size(1)
    target = target * (1 - smoothing) + smoothing / k
    return -(target * F.log_softmax(logits.float(), 1)).sum(1).mean()


class MixStyle(nn.Module):
    """In training, with probability p, each image's feature statistics (channel mean and
    standard deviation) are mixed with another image's, weight drawn from Beta(alpha, alpha).
    Identity at evaluation; no parameters, so saved weights are unaffected."""

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = p, alpha, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or x.size(0) < 2 or torch.rand(1).item() >= self.p:
            return x
        mu = x.mean((2, 3), keepdim=True).detach()
        sig = (x.var((2, 3), keepdim=True) + self.eps).sqrt().detach()
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((x.size(0), 1, 1, 1)).to(x)
        perm = torch.randperm(x.size(0), device=x.device)
        return (x - mu) / sig * (lam * sig + (1 - lam) * sig[perm]) + lam * mu + (1 - lam) * mu[perm]


def sam_step(model: nn.Module, loss_of, opt, scaler, rho: float, autocast) -> tuple[torch.Tensor, torch.Tensor]:
    """One SAM step. loss_of(model) -> (outputs, loss) for the current batch; autocast is the
    context to run it under. The first pass gives the gradient g; the weights move rho * g/|g|
    (the scaler's loss scale cancels in the direction), the second pass's gradient is applied to
    the original weights. Batch norm's running statistics are updated by the first pass only."""
    with autocast():
        out, loss = loss_of(model)
    scaler.scale(loss).backward()
    params = [p for p in model.parameters() if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.float()) for p in params]))
    if torch.isfinite(norm) and norm > 0:
        with torch.no_grad():
            eps = [p.grad * (rho / norm) for p in params]
            torch._foreach_add_(params, eps)
        opt.zero_grad(set_to_none=True)
        bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        momenta = [m.momentum for m in bns]
        for m in bns:
            m.momentum = 0.0
        with autocast():
            _, loss2 = loss_of(model)
        scaler.scale(loss2).backward()
        for m, mom in zip(bns, momenta):
            m.momentum = mom
        with torch.no_grad():
            torch._foreach_sub_(params, eps)
    scaler.step(opt)
    scaler.update()
    return out, loss
