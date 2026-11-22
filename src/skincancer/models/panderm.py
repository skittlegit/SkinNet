"""PanDerm's ViT encoder (Yan et al., Nature Medicine 2025) with its published skin-image weights
(DECISIONS.md, sections 23 and 24). The weights are CC BY-NC-ND 4.0: kept in data/weights/,
never redistributed."""
from __future__ import annotations

import torch.nn as nn


PANDERM_BASE_WEIGHTS = "data/weights/panderm_base.pth"   # PanDerm_Base, github.com/SiyuanYan1/PanDerm (CC BY-NC-ND 4.0)
# PanDerm (ViT-L/16, the paper's network), same source and licence: the `encoder.` part of the
# published pretraining checkpoint, saved with that prefix removed
PANDERM_LARGE_WEIGHTS = "data/weights/panderm_large.pth"


class PanDermBase(nn.Module):
    """PanDerm's ViT-B/16 encoder (Yan et al., Nature Medicine 2025), built with timm's BEiT,
    whose layout its checkpoint matches key for key: layer scale, q/v biases, fixed 2-D sin-cos
    position embedding (kept frozen, as in PanDerm), no relative position bias. ImageNet
    normalisation, the class token after the final norm: 768-d embedding. Pretrained at
    224 px; for another img_size (idea PB) the position embedding is resampled to the new
    grid of patches. DECISIONS.md, sections 23 and 24. pretrained=False builds it with
    random weights (used when a saved best.pt is loaded next)."""

    def __init__(self, pretrained: bool = False, path: str | None = None, img_size: int = 224, large: bool = False,
                 drop_path: float = 0.0):
        """large=True: PanDerm's ViT-L/16 (24 blocks, 1024-d), idea PC. drop_path: stochastic
        depth, each block skipped at random in training with a rate rising to this at the last
        block (idea PE; PanDerm's own fine-tuning uses 0.2)."""
        super().__init__()
        import timm
        import torch
        path = path or (PANDERM_LARGE_WEIGHTS if large else PANDERM_BASE_WEIGHTS)
        self.net = timm.create_model("beit_large_patch16_224" if large else "beit_base_patch16_224", pretrained=False, num_classes=0, global_pool="token",
                                     use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                                     init_values=0.1, img_size=img_size, drop_path_rate=drop_path)
        if pretrained:
            sd = torch.load(path, map_location="cpu", weights_only=True)
            if img_size != 224:
                from timm.layers import resample_abs_pos_embed
                sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], (img_size // 16, img_size // 16),
                                                         num_prefix_tokens=1)
            missing, unexpected = self.net.load_state_dict(sd, strict=False)
            if unexpected or [k for k in missing if not k.endswith("k_bias")]:
                raise ValueError(f"PanDerm weights do not fit: missing {missing}, unexpected {unexpected}")
        self.net.pos_embed.requires_grad_(False)
        self.num_features = self.net.num_features

    def forward(self, x):
        return self.net(x)
