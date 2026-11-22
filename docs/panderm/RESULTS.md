# Results: PanDerm track

PanDerm's pretrained ViT fine-tuned on our data (DECISIONS.md, sections 23 and 24). Every number is the mean of three seeds with a 95% confidence interval unless stated. The rules are in [DECISIONS.md](../../DECISIONS.md). Both tracks side by side: [COMPARISON.md](../COMPARISON.md); the from-scratch track: [SkinNet results](../skinnet/RESULTS.md).

## PanDerm track, idea PF: a running average of the weights (20 Nov 2026)

PanDerm checkpoint 1's recipe, validated, selected and kept as an exponential average of its weights (decay 0.999 per step). Three seeds, test-time flips and the seed average; passed step 1, then scored once on every set.

| Set | Score | PanDerm checkpoint 1 | Idea PF | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `val` | Within-source AUC | 0.983 | 0.985 | +0.002 [−0.002, +0.006] | no |
| `midas_val` | AUC | 0.856 | 0.872 | +0.016 [−0.002, +0.032] | no |
| `test` | AUC | 0.977 | 0.978 | +0.001 [−0.002, +0.005] | no |
| `test` | Within-source AUC | 0.982 | 0.982 | +0.000 [−0.004, +0.003] | no |
| `midas_test` | AUC | 0.874 | 0.874 | −0.000 [−0.016, +0.018] | no |
| DDI | AUC | 0.686 | 0.668 | −0.018 [−0.031, −0.006] | **yes, lower** |
| DermaCon-IN (65 photos) | AUC | 0.850 | 0.857 | +0.007 [−0.011, +0.036] | no |

**Outcome: discarded.** Level at home and on MIDAS test, lower on DDI by a supported amount (DECISIONS.md, section 24).

## PanDerm track, idea PC: PanDerm Large, PanDerm checkpoint 1 (17 Nov 2026)

PA's recipe on PanDerm's ViT-L/16 (304 million parameters) in place of the ViT-B/16. Three seeds, test-time flips and the seed average, scored once on every set.

| Set | Score | PanDerm base | Idea PA | PanDerm Large (PC) | PC − base [95% interval] | PC − PA [95% interval] |
| --- | --- | --- | --- | --- | --- | --- |
| `val` | Within-source AUC | 0.967 | 0.975 | **0.983** | +0.016 [+0.008, +0.025] | +0.009 [+0.003, +0.015] |
| `midas_val` | AUC | 0.835 | 0.842 | 0.856 | +0.022 [−0.007, +0.052] | +0.014 [−0.014, +0.045] |
| `test` | AUC | 0.968 | 0.967 | **0.977** | +0.010 [+0.003, +0.016] | +0.011 [+0.003, +0.018] |
| `test` | Within-source AUC | 0.971 | 0.970 | **0.982** | +0.010 [+0.004, +0.018] | +0.012 [+0.006, +0.020] |
| `midas_test` | AUC | 0.811 | 0.848 | **0.874** | +0.063 [+0.032, +0.097] | +0.026 [−0.003, +0.057] |
| DDI | AUC | 0.650 | 0.650 | **0.686** | +0.036 [+0.007, +0.064] | +0.035 [+0.011, +0.059] |
| DermaCon-IN (65 photos) | AUC | 0.880 | 0.933 | 0.850 | −0.030 [−0.164, +0.062] | −0.082 [−0.220, +0.012] |

Against checkpoint 4 (from scratch):

| Set | Score | Checkpoint 4 (from scratch) | PanDerm checkpoint 1 | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `test` | Within-source AUC | 0.907 | 0.982 | +0.074 [+0.056, +0.096] | yes |
| `midas_test` | AUC | 0.750 | 0.874 | +0.124 [+0.074, +0.174] | yes |
| DDI | AUC | 0.650 | 0.686 | +0.036 [−0.010, +0.081] | no |
| DermaCon-IN (65 photos) | AUC | 0.818 | 0.850 | +0.032 [−0.123, +0.181] | no |

**Outcome: passes the rule against both earlier PanDerm runs; PanDerm checkpoint 1.** DDI rises for the first time (0.650 → 0.686, supported). DermaCon-IN is 0.082 lower than PA's, not supported on 65 photos.

## PanDerm track, idea PD: colour augmentation (17 Nov 2026)

PA's recipe with skin tone and lighting colour varied in training. Against PA, validation only:

| Set | Score | Idea PA | Idea PD | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` | AUC | 0.842 | 0.835 | −0.007 [−0.032, +0.018] | no |
| `val` | AUC | 0.972 | 0.972 | +0.000 [−0.004, +0.005] | no |
| `val` | Within-source AUC | 0.975 | 0.976 | +0.002 [−0.003, +0.007] | no |

**Outcome: discarded at the first step**, not scored on any test set.

AUC on `val` by Fitzpatrick skin type (validation only):

| Model | Types I–III (1,801 photos, 377 cancers) | Type IV (419, 46) | Types V–VI (342, 35) |
| --- | --- | --- | --- |
| Checkpoint 4 | 0.901 | 0.888 | 0.895 |
| PanDerm base | 0.965 | 0.952 | 0.963 |
| Idea PA | 0.969 | 0.976 | 0.988 |
| Idea PD | 0.971 | 0.970 | 0.982 |

**Reading.** At home, no model is worse on types V–VI than on I–III (35 cancers, so the figures are rough). DDI's low score is therefore not a matter of dark skin alone; what differs there is the hospital, the kind of photo and the mix of diagnoses.

## PanDerm track, idea PA: gentler fine-tuning (17 Nov 2026)

PanDerm's ViT-B/16 on checkpoint 4's data, the head trained alone for 3 epochs, then every layer for 20 with a learning rate that shrinks by 0.75 per block towards the input. Three seeds, test-time flips and the seed average, scored once on every set. Against the first PanDerm run (`panderm`, every layer at 5e-5 from the first step):

| Set | Score | PanDerm base | Idea PA | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `val` | Within-source AUC | 0.967 | 0.975 | +0.007 [+0.000, +0.015] | yes, a gain |
| `midas_val` | AUC | 0.835 | 0.842 | +0.007 [−0.018, +0.034] | no |
| `test` | AUC | 0.968 | 0.967 | −0.001 [−0.007, +0.005] | no |
| `test` | Within-source AUC | 0.971 | 0.970 | −0.002 [−0.008, +0.004] | no |
| `midas_test` | AUC | 0.811 | 0.848 | +0.037 [+0.008, +0.067] | yes, a gain |
| DDI | AUC | 0.650 | 0.650 | +0.001 [−0.024, +0.026] | no |
| DermaCon-IN (65 photos) | AUC | 0.880 | 0.933 | +0.053 [+0.000, +0.122] | no |

Against checkpoint 4 (from scratch): `test` within-source AUC +0.062, MIDAS test 0.750 → 0.848 (+0.098), DermaCon-IN 0.818 → 0.933 (+0.114), all supported; DDI 0.650 → 0.650, level.

**Outcome: a narrow miss of the rule, with the team.** The only supported difference on a test set is the gain on MIDAS test; `test` is 0.002 lower, not supported; DDI did not move.

## PanDerm reference (16 Nov 2026)

PanDerm's ViT-B/16, pretrained by its authors on about 2.1 million skin images, trained on checkpoint 4's data (learning rate 5e-5, 20 epochs). Three seeds, test-time flips and the seed average, scored once on every set. A reference like idea P, never a checkpoint. Against checkpoint 4:

| Set | Score | Checkpoint 4 | PanDerm | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `val` | AUC | 0.900 | 0.964 | +0.065 [+0.051, +0.079] | yes, a gain |
| `val` | Within-source AUC | 0.905 | 0.967 | +0.062 [+0.045, +0.080] | yes, a gain |
| `midas_val` | AUC | 0.818 | 0.835 | +0.016 [−0.030, +0.060] | no |
| `test` | AUC | 0.903 | 0.968 | +0.065 [+0.051, +0.079] | yes, a gain |
| `test` | Within-source AUC | 0.907 | 0.971 | +0.064 [+0.047, +0.083] | yes, a gain |
| `midas_test` | AUC | 0.750 | 0.811 | +0.061 [+0.004, +0.113] | yes, a gain |
| DDI | AUC | 0.650 | 0.650 | −0.000 [−0.045, +0.042] | no |
| DermaCon-IN (65 photos) | AUC | 0.818 | 0.880 | +0.062 [−0.009, +0.154] | no |

Against the ImageNet ResNet-18 at learning rate 1e-4:

| Set | Score | ImageNet ResNet-18 (lr 1e-4) | PanDerm | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `val` | Within-source AUC | 0.961 | 0.967 | +0.006 [−0.002, +0.014] | no |
| `midas_val` | AUC | 0.778 | 0.835 | +0.057 [+0.017, +0.098] | yes, a gain |
| `test` | Within-source AUC | 0.964 | 0.971 | +0.007 [+0.000, +0.015] | yes, a gain |
| `midas_test` | AUC | 0.737 | 0.811 | +0.074 [+0.033, +0.113] | yes, a gain |
| DDI | AUC | 0.631 | 0.650 | +0.019 [−0.018, +0.055] | no |
| DermaCon-IN (65 photos) | AUC | 0.804 | 0.880 | +0.076 [−0.087, +0.281] | no |

**Reading.** PanDerm is the first pretrained network to be ahead of checkpoint 4 outside as well as at home: MIDAS test +0.061, supported, though the interval nearly reaches zero and MIDAS validation's gain (+0.016) is not supported. On DDI it is exactly level with checkpoint 4 (0.650): two million skin images in pretraining did nothing for the set with the darkest skin. Against the ImageNet ResNet-18 it is level at home and clearly better on MIDAS, so what skin pretraining adds over ImageNet is outside performance on clinic photos. The home gain (+0.06) is the one most exposed to PanDerm having seen atlas photos in pretraining (DECISIONS.md, section 23); MIDAS and DDI are not named among its data.
