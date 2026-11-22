"""The two tracks' networks behind one classifier head.

- `model.arch: skinnet` (models/skinnet.py): our own convolutional network, from random weights.
- `model.arch: panderm_base | panderm_large` (models/panderm.py): PanDerm's ViT-B/16 or ViT-L/16,
  `model.weights: panderm` for its published weights.

A torchvision ResNet-18 reference (idea P, DECISIONS.md, section 19) was removed on 23 Nov 2026;
its code is at git tag pre-restructure."""
from __future__ import annotations

import torch.nn as nn

from .panderm import PANDERM_BASE_WEIGHTS, PANDERM_LARGE_WEIGHTS, PanDermBase
from .skinnet import ResBlock, SkinNet

__all__ = ["Classifier", "PanDermBase", "PANDERM_BASE_WEIGHTS", "PANDERM_LARGE_WEIGHTS", "ResBlock", "SkinNet",
           "target_layer"]


class Classifier(nn.Module):
    """Backbone (SkinNet unless arch says otherwise) + dropout + one linear layer.
    pretrained loads the arch's outside weights (PanDerm's; SkinNet has none)."""

    def __init__(self, num_classes: int, width: int = 32, depths=(2, 2, 2, 2), drop: float = 0.3,
                 arch: str = "skinnet", pretrained: bool = False, img_size: int = 224, drop_path: float = 0.0):
        super().__init__()
        if arch == "skinnet":
            if pretrained:
                raise ValueError("SkinNet has no outside weights")
            self.backbone = SkinNet(width, depths)
        elif arch in ("panderm_base", "panderm_large"):
            self.backbone = PanDermBase(pretrained, img_size=img_size, large=arch == "panderm_large",
                                        drop_path=drop_path)
        else:
            raise ValueError(f"unknown model.arch {arch!r}")
        self.dim = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(self.dim, num_classes))

    def forward(self, x, return_features: bool = False):
        f = self.backbone(x)
        logits = self.head(f)
        return (logits, f) if return_features else logits


def target_layer(model: Classifier) -> nn.Module:
    """Last residual block, for Grad-CAM."""
    if isinstance(model.backbone, PanDermBase):
        raise ValueError("Grad-CAM is set up for convolutional networks only")
    return model.backbone.stages[-1][-1]
