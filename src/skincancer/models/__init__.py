"""SkinNet behind one classifier head.

- `model.arch: skinnet` (models/skinnet.py): our own convolutional network, from random weights.

PanDerm's ViTs (`model.arch: panderm_base | panderm_large`) moved to their own repository,
skittlegit/PanDerm, on 24 Nov 2026 (DECISIONS.md, section 31). A torchvision ResNet-18
reference (idea P, section 19) was removed on 23 Nov 2026; its code is at git tag pre-restructure."""
from __future__ import annotations

import torch.nn as nn

from .skinnet import ResBlock, SkinNet

__all__ = ["Classifier", "ResBlock", "SkinNet", "target_layer"]


class Classifier(nn.Module):
    """SkinNet + dropout + one linear layer. SkinNet has no outside weights."""

    def __init__(self, num_classes: int, width: int = 32, depths=(2, 2, 2, 2), drop: float = 0.3,
                 arch: str = "skinnet", pretrained: bool = False, img_size: int = 224, mixstyle: dict | None = None):
        super().__init__()
        if arch != "skinnet":
            raise ValueError(f"unknown model.arch {arch!r} (PanDerm's ViTs are in skittlegit/PanDerm)")
        if pretrained:
            raise ValueError("SkinNet has no outside weights")
        self.backbone = SkinNet(width, depths, mixstyle)
        self.dim = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(self.dim, num_classes))

    def forward(self, x, return_features: bool = False):
        f = self.backbone(x)
        logits = self.head(f)
        return (logits, f) if return_features else logits


def target_layer(model: Classifier) -> nn.Module:
    """Last residual block, for Grad-CAM."""
    return model.backbone.stages[-1][-1]
