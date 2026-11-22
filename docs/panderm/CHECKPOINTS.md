# Checkpoints: PanDerm track, earlier ones

The current models of both tracks are in [CHECKPOINTS.md](../CHECKPOINTS.md). Details in [RESULTS.md](RESULTS.md).

## PanDerm checkpoint 1 (17 Nov 2026)

Since 17 Nov there are two tracks (DECISIONS.md, section 24): the from-scratch one (then at checkpoint 4, below), and the PanDerm one, which starts from PanDerm's pretrained weights (Yan et al., Nature Medicine 2025; CC BY-NC-ND 4.0, non-commercial, not redistributed). One of the two is picked at the end.

PanDerm's ViT-L/16, the network of its paper, fine-tuned on checkpoint 4's training data with our recipe (idea PA): the head trained alone for 3 epochs, then every layer with a learning rate that shrinks by 0.86 per block towards the input; early stopping 4 epochs after the best.

| | |
| --- | --- |
| Training data | The 18,022 images of checkpoint 4 |
| Training | 3 epochs head only, then up to 20; about 10 minutes an epoch with gradient checkpointing, about 2 hours a seed |
| Config | `configs/panderm/panderm-c1.yaml` (weights `data/weights/panderm_large.pth`, local) |
| Run folder | `runs/panderm-c1`: seeds in `s0`, `s1`, `s2`, their average in `avg` |
| Weights | `checkpoints/panderm-c1/` (local, git-ignored) |
| Git tag | `panderm-checkpoint-1` |

| Set | Score | PanDerm base | Idea PA | PanDerm Large (PC) | PC − base [95% interval] | PC − PA [95% interval] |
| --- | --- | --- | --- | --- | --- | --- |
| `val` | Within-source AUC | 0.967 | 0.975 | **0.983** | +0.016 [+0.008, +0.025] | +0.009 [+0.003, +0.015] |
| `midas_val` | AUC | 0.835 | 0.842 | 0.856 | +0.022 [−0.007, +0.052] | +0.014 [−0.014, +0.045] |
| `test` | AUC | 0.968 | 0.967 | **0.977** | +0.010 [+0.003, +0.016] | +0.011 [+0.003, +0.018] |
| `test` | Within-source AUC | 0.971 | 0.970 | **0.982** | +0.010 [+0.004, +0.018] | +0.012 [+0.006, +0.020] |
| `midas_test` | AUC | 0.811 | 0.848 | **0.874** | +0.063 [+0.032, +0.097] | +0.026 [−0.003, +0.057] |
| DDI | AUC | 0.650 | 0.650 | **0.686** | +0.036 [+0.007, +0.064] | +0.035 [+0.011, +0.059] |
| DermaCon-IN (65 photos) | AUC | 0.880 | 0.933 | 0.850 | −0.030 [−0.164, +0.062] | −0.082 [−0.220, +0.012] |

Against checkpoint 4:

| Set | Score | Checkpoint 4 (from scratch) | PanDerm checkpoint 1 | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `test` | Within-source AUC | 0.907 | 0.982 | +0.074 [+0.056, +0.096] | yes |
| `midas_test` | AUC | 0.750 | 0.874 | +0.124 [+0.074, +0.174] | yes |
| DDI | AUC | 0.650 | 0.686 | +0.036 [−0.010, +0.081] | no |
| DermaCon-IN (65 photos) | AUC | 0.818 | 0.850 | +0.032 [−0.123, +0.181] | no |

**Where it stands.** The first model to raise DDI by a supported amount (over the earlier PanDerm runs), and far ahead of checkpoint 4 on MIDAS test. DermaCon-IN is lower than with PA by a large but unsupported amount (65 photos). Its home scores may be inflated by atlas photos in PanDerm's pretraining (DECISIONS.md, section 23).
