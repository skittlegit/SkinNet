# Checkpoints

A checkpoint is a model we keep and build on. Each idea starts from the latest one and has to beat it to become the next ([DECISIONS.md](../DECISIONS.md), section 15 for names; section 26 for the rule). Every idea by letter, kept and discarded, is in [IDEAS.md](IDEAS.md); full numbers in [RESULTS.md](RESULTS.md).

**The current model is checkpoint 5 (`skinnet-c5`, 25 Nov 2026):** checkpoint 4 trained with sharpness-aware minimisation and a running average of its weights (was idea Y; DECISIONS.md, sections 32 and 33). New ideas are compared with it.

| Model | Name | Built from | Weights (local) |
| --- | --- | --- | --- |
| Baseline | `skinnet` | `configs/skinnet.yaml` | |
| Checkpoint 1 | `skinnet-c1` | `configs/skinnet-c1.yaml` (was idea A) | `checkpoints/skinnet-c1/` |
| Checkpoint 2, first step (12 Nov) | `skinnet-c1-tta` | checkpoint 1's three seeds with test-time flips, averaged (ideas G + H); nothing retrained | `checkpoints/skinnet-c1-tta/` |
| Checkpoint 2 (13 Nov) | `skinnet-c2` | `configs/skinnet-c2.yaml` (was idea O): the first step plus MIDAS photos in training | `checkpoints/skinnet-c2/` |
| Checkpoint 3 | `skinnet-c3` | `configs/skinnet-c3.yaml` (was idea Q) | `checkpoints/skinnet-c3/` |
| Checkpoint 4 | `skinnet-c4` | `configs/skinnet-c4.yaml` (was idea U) | `checkpoints/skinnet-c4/` |
| Checkpoint 4, all-DDI version | `skinnet-idea-t` | `configs/skinnet-idea-t.yaml`: checkpoint 4 with `ddi_val` also in training | `checkpoints/skinnet-idea-t/` |
| **Checkpoint 5** | `skinnet-c5` | `configs/skinnet-c5.yaml` (was idea Y) | `checkpoints/skinnet-c5/` |

One name per model, the same for its config (`configs/<name>.yaml`), run folder (`runs/<name>/`, seeds `s0`–`s2` and their average `avg`) and result folder (`results/<name>/metrics.csv`). An idea being tested is `skinnet-idea-<letter>`; if kept, `scripts/adopt_checkpoint.py` renames it to the next checkpoint everywhere.

Each section below gives a checkpoint's recipe, its scores against the checkpoint before it, and how it was adopted, newest first. Scores are seed averages with test-time flips; paired differences have 95% intervals, and "supported" means the interval excludes zero. Its scores on every test set are in the [README](../README.md); details in [RESULTS.md](RESULTS.md).

## Checkpoint 5 (25 Nov 2026)

Checkpoint 4 trained for flat minima (idea Y): sharpness-aware minimisation (SAM, rho 0.05), which takes each step against the worst nearby weights, plus a running average of the weights (decay 0.9995 per step, batch norm statistics included). Nothing else changed.

| | |
| --- | --- |
| Training data | 18,186 images, as checkpoint 4 |
| Training | 60 epochs, early stopping off; SAM takes two passes per step, so about an hour per seed |
| Config | `configs/skinnet-c5.yaml` |
| Run folder | `runs/skinnet-c5`: seeds in `s0`, `s1`, `s2`, their average in `avg` |
| Weights | `checkpoints/skinnet-c5/` (local, git-ignored) |
| Git tag | `checkpoint-5` |

Step 1, validation sets: higher than checkpoint 4 on all three, none by a supported amount. It was the first SkinNet training change to pass step 1.

| Set | Checkpoint 4 | Checkpoint 5 | Paired difference [95% interval] |
| --- | --- | --- | --- |
| `val`, within-source AUC | 0.903 | 0.906 | +0.003 [−0.005, +0.011] |
| `midas_val` | 0.808 | 0.815 | +0.007 [−0.009, +0.024] |
| `ddi_val` | 0.749 | 0.755 | +0.006 [−0.023, +0.036] |

Step 2, test sets, scored once:

| Set | Checkpoint 4 | Checkpoint 5 | Paired difference [95% interval] |
| --- | --- | --- | --- |
| Our test split | 0.901 | 0.901 | −0.000 [−0.006, +0.006] |
| MIDAS test | 0.744 | 0.734 | −0.010 [−0.034, +0.012] |
| DDI, held-out half | 0.670 | 0.676 | +0.006 [−0.018, +0.029] |
| DermaCon-IN tumours | 0.847 | 0.846 | −0.001 [−0.030, +0.028] |
| HIBA | 0.860 | 0.850 | −0.011 [−0.028, +0.008] |
| MED-NODE | 0.771 | **0.823** | **+0.052 [+0.012, +0.094]** |
| MCR-SL | 0.804 | 0.805 | +0.001 [−0.031, +0.032] |
| Derm7pt | 0.714 | 0.712 | −0.002 [−0.016, +0.013] |
| Asan and Hallym | 0.784 | **0.793** | **+0.009 [+0.005, +0.012]** |
| ISIC 2019 (dermoscopy) | **0.750** | 0.729 | **−0.021 [−0.025, −0.016]** |

**Where it stands.** The first change to training, rather than to the data, that gained on outside test sets: MED-NODE and Asan, both supported. It lost on ISIC 2019, the one dermoscopy set; SkinNet learns from clinical photos, and idea V lost there too. Against the base model `skinnet` on the validation sets, the five checkpoints together are worth `val` +0.052, `midas_val` +0.153 and `ddi_val` +0.070, all supported (DECISIONS.md, section 35).

**How it became a checkpoint.** The rule written before its test scores (DECISIONS.md, section 32) asked for nothing lower by a supported amount; ISIC 2019 was. The user adopted it on 25 Nov as a narrow miss (section 33).

## Checkpoint 4 (22 Nov 2026)

Checkpoint 3's recipe with a quarter of DDI (`ddi_train`, 164 photos) added to training (idea U). Another quarter (`ddi_val`, 164) became a DDI validation set, used in step 1 from then on; the other half (`ddi_hold`, 328) stays test only (DECISIONS.md, section 26).

| | |
| --- | --- |
| Training data | 18,186 images: the 18,022 of checkpoint 3 plus 164 from DDI |
| Training | 60 epochs, early stopping off, about 30 minutes per seed |
| Config | `configs/skinnet-c4.yaml` |
| Run folder | `runs/skinnet-c4`: seeds in `s0`, `s1`, `s2`, their average in `avg` |
| Weights | `checkpoints/skinnet-c4/` (local, git-ignored) |
| Git tag | `checkpoint-4` |

| Set | Checkpoint 3 | Checkpoint 4 | Paired difference [95% interval] |
| --- | --- | --- | --- |
| `val`, within-source AUC (step 1) | 0.905 | 0.903 | −0.003 [−0.012, +0.006] |
| `midas_val` (step 1) | 0.818 | 0.808 | −0.010 [−0.030, +0.010] |
| `ddi_val` (step 1) | 0.701 | **0.749** | **+0.048 [+0.015, +0.087]** |
| Our test split | 0.903 | 0.901 | −0.002 [−0.007, +0.003] |
| MIDAS test | 0.750 | 0.744 | −0.006 [−0.026, +0.012] |
| DDI, held-out half | 0.641 | **0.670** | **+0.029 [+0.007, +0.052]** |
| DermaCon-IN tumours | 0.818 | 0.847 | +0.029 [−0.014, +0.084] |
| MED-NODE | **0.822** | 0.771 | **−0.051**, supported |
| Asan and Hallym | 0.775 | **0.784** | **+0.009**, supported |

HIBA, MCR-SL, Derm7pt and ISIC 2019 were level (section 27).

**Where it stands.** 164 DDI photos raise DDI's held-out half by 0.029, two thirds of the gain from 328 (idea T, +0.043), and the DDI validation set agrees with the test half on the direction and size of the gain. Home and MIDAS are level. DDI is no longer a hospital the model has never seen: its never-seen DDI score is checkpoint 3's (0.650 on all of DDI).

**How it became a checkpoint.** It failed step 1 (`val` and `midas_val` a little lower) and was not higher on every test set. The user chose it on 22 Nov for its DDI validation set, so that every later idea can be checked on DDI before a test set is looked at (section 26).

**The all-DDI version** (`skinnet-idea-t`, tag `checkpoint-4-all-ddi`): the same recipe with `ddi_val` also in training, 328 DDI photos. `ddi_hold` 0.684, level with checkpoint 4. It cannot be checked on `ddi_val`, so new ideas are compared with checkpoint 4's line, not with it.

## Checkpoint 3 (14 Nov 2026)

Checkpoint 2's recipe with 4,908 biopsy-confirmed clinical close-ups from MILK10k added to training (idea Q): cancers and the benign growths that resemble them, from five centres not otherwise in the data (Vienna, Ankara, Queensland, Memorial Sloan Kettering, Skopje). Predicts the same way (test-time flips, three-seed average).

| | |
| --- | --- |
| Training data | 18,022 images: the 13,114 of checkpoint 2 plus 4,908 from MILK10k (3,620 cancer, 1,288 not cancer; training only) |
| Training | 60 epochs, early stopping off, about 30 minutes per seed |
| Config | `configs/skinnet-c3.yaml` |
| Run folder | `runs/skinnet-c3`: seeds in `s0`, `s1`, `s2`, their average in `avg` |
| Weights | `checkpoints/skinnet-c3/` (local, git-ignored) |
| Git tag | `checkpoint-3` |

| Score | Checkpoint 2 | Checkpoint 3 | Paired difference [95% interval] |
| --- | --- | --- | --- |
| MIDAS test part (414 images): AUC | 0.698 | **0.750** | +0.052 [+0.004, +0.098] |
| DDI (outside, never seen): AUC | 0.657 | 0.650 | −0.007 [−0.034, +0.021] |
| DermaCon-IN tumours (outside): AUC | 0.841 | 0.818 | −0.023 [−0.088, +0.030] |
| Our test split: AUC | 0.898 | 0.903 | +0.005 [−0.003, +0.013] |
| Our test split: within-source AUC | 0.876 | **0.907** | +0.031 [+0.014, +0.052] |
| Our test split: AUC, cancers vs benign lookalikes | 0.775 | **0.831** | |
| Our test split: cancers missed / false alarms | 10% / 34% | 9% / 33% | |
| MIDAS test: cancers missed / false alarms | 33% / 41% | **29% / 29%** | |
| DDI: cancers missed / false alarms | 16% / 67% | 14% / 69% | |
| DermaCon-IN everyday conditions called cancer | 28% | 31% | |
| Actinic keratoses called cancer (never trained on) | 80% | 88% | |

**Where it stands.** The best model so far on the hard question, cancer against a benign growth: up 0.056 at home on lookalikes and 0.052 on MIDAS test, with false alarms on MIDAS test down from 41% to 29%. On DDI it is flat, and not because of skin tone: by DDI's skin-tone groups it is 0.676 → 0.645 on I–II, 0.679 → 0.713 on III–IV and 0.625 → 0.607 on V–VI, about 200 images each, all within noise. Why MILK10k helps on MIDAS and not on DDI is open. It calls actinic keratoses (pre-cancers, never trained on) cancer more often, which fits its training: half of MILK10k is basal cell carcinoma.

**How it became a checkpoint.** The rule written before its test scores (DECISIONS.md, section 19) asked for a rise on every test set. It rose on our test split and MIDAS test, both supported, and was 0.007 lower on DDI, well inside the noise; nothing was lower by a supported amount. The team accepted it on 14 Nov, after seeing the result, as a narrow miss of the kind section 14 leaves to the team. Recorded here as such.

**From now on** every new idea is compared with checkpoint 3. Idea L, specified against checkpoint 2 before checkpoint 3 existed, was judged against checkpoint 2 as written, and failed (DECISIONS.md, section 17).

## Checkpoint 2 (12–13 Nov 2026)

Checkpoint 2 combines two steps taken a day apart. First, checkpoint 1's three models were averaged, each predicting on the photo and its mirror images (12 Nov). Then checkpoint 1's recipe was trained again with 552 MIDAS photos added, predicting the same way (13 Nov). Until 26 Nov the first step was counted as a checkpoint of its own; the numbers were then merged (DECISIONS.md, section 37).

### Second step: MIDAS photos in training (13 Nov)

Checkpoint 1's recipe trained again with photos from a second kind of source: 552 biopsy-confirmed hospital photos from MIDAS. Predicts the same way as the first step (test-time flips, three-seed average).

| | |
| --- | --- |
| Training data | 13,114 images: the 12,562 of checkpoint 1 plus 552 from MIDAS (`midas_train`) |
| Training | 60 epochs, early stopping off, about 20 minutes per seed |
| Config | `configs/skinnet-c2.yaml` |
| Run folder | `runs/skinnet-c2`: seeds in `s0`, `s1`, `s2`, their average in `avg` |
| Weights | `checkpoints/skinnet-c2/` (local, git-ignored) |
| Git tag | `checkpoint-2` |

| Score | First step | Checkpoint 2 | Paired difference [95% interval] |
| --- | --- | --- | --- |
| DDI (outside, never seen): AUC | 0.629 | **0.657** | +0.028 [+0.007, +0.048] |
| MIDAS test part (414 images): AUC | 0.645 | **0.698** | +0.053 [+0.020, +0.089] |
| DermaCon-IN tumours (outside): AUC | 0.826 | 0.841 | +0.015 [−0.058, +0.090] |
| Our test split: AUC | **0.904** | 0.898 | −0.007 [−0.012, −0.001] |
| Our test split: AUC, cancers vs benign lookalikes | **0.797** | 0.775 | |
| Our test split: cancers missed / false alarms | 9% / 33% | 10% / 34% | |
| DDI: cancers missed / false alarms | 21% / 59% | 16% / 67% | |
| MIDAS test: cancers missed / false alarms | 50% / 31% | 33% / 41% | |
| DermaCon-IN everyday conditions called cancer | **24%** | 28% | |

**Where it stands.** It is the first model here shown to do better on a source it has never seen: DDI. The MIDAS gain is larger, but the model has now seen that hospital's photos, so it says less. It costs a little at home. The threshold fixed on our own validation split still does not carry over: outside, fewer cancers are missed but there are more false alarms.

**How it became a checkpoint.** The rule written beforehand (DECISIONS.md, sections 11 and 12) asked for a supported gain on MIDAS test or DDI, which it has on both, and no supported loss on our own test split, which it missed by a small margin (−0.007). The team decided the trade is worth taking, because performance on new hospitals is the open problem. That is a decision made after seeing the result, recorded here as such.

**From now on** every new idea is compared with checkpoint 2 after the same treatment (three seeds, 60 epochs, test-time flips, seed average), and chosen on the MIDAS validation part as well as our own.

### First step: averaging seeds and mirror images (12 Nov; `skinnet-c1-tta`)

Checkpoint 1's three trained models, used better when predicting. Nothing was retrained.

| | |
| --- | --- |
| What it is | For each image, each of the three seeds predicts on the image and its three mirror images (left-right, up-down, both); the twelve cancer probabilities are averaged into one |
| Weights | The same as checkpoint 1; copies with the prediction settings are in `checkpoints/skinnet-c1-tta/` (local, git-ignored) |
| Run folder | `runs/skinnet-c1-tta`: seeds in `s0`, `s1`, `s2`, their average in `avg` |
| Git tag | `checkpoint-1-tta` |
| Built with | `python -m skincancer.derive` (commands below) |

| Score | Baseline | Checkpoint 1 | First step |
| --- | --- | --- | --- |
| Our test split: AUC | 0.887 | 0.890 | **0.904** |
| Our test split: AUC, cancers vs benign lookalikes | 0.723 | 0.783 | **0.797** |
| Cancers missed | 9% | 8% | 9% |
| False alarms, all non-cancers | 42% | 40% | **33%** |
| False alarms, PAD-UFES-20 non-cancers | 89% | 63% | 57% |
| False alarms, benign lookalikes | 81% | 65% | 61% |
| MIDAS (outside): AUC | 0.603 | 0.628 | 0.634 |
| DDI (outside): AUC | 0.654 | 0.623 | 0.629 |
| DermaCon-IN tumours (outside): AUC | 0.825 | 0.793 | 0.826 |
| DermaCon-IN everyday conditions called cancer | 35% | 34% | **24%** |

Missed cancers and false alarms use the threshold that catches 90% of cancers on the validation split.

**Where it stands.** On photos like its training data it is a clear step up from checkpoint 1: +0.015 AUC and a sixth fewer false alarms, a gain the paired comparison supports. On photos from other hospitals it is not shown to be better: MIDAS and DDI each moved by +0.006, which cannot be told from no change. The outside gap is still the open problem.

**How it was kept.** The rule written before the test (DECISIONS.md, section 10) asked for a supported gain on MIDAS or DDI, and by that rule it was first discarded. On seeing the result the team decided that a free, supported gain at home with nothing worse anywhere counts as working, and restored it. That was a change of rule after the result, recorded here so the outside numbers are not read as better than they are.

**Until the second step** every new idea was compared with the first step, and got the same treatment before the comparison: three seeds, test-time flips, seed average.

## Checkpoint 1 (12 Nov 2026)

| | |
| --- | --- |
| Task | Cancer or not cancer, from a clinical skin photo |
| Network | SkinNet: our own residual CNN, 2.8M parameters, trained from random weights |
| Training data | 12,562 images (2,220 cancer) from Fitzpatrick17k and PAD-UFES-20 |
| Training | 60 epochs at 224 × 224, AdamW, about 20 minutes per seed on an RTX 4060 |
| What sets it apart from the baseline | Inside each dataset, cancer and not cancer are drawn equally often, so the kind of photo no longer predicts the class |
| Config | `configs/skinnet-c1.yaml` |
| Git tag | `checkpoint-1` |
| Weights | `checkpoints/skinnet-c1/s{0,1,2}/best.pt` (local, git-ignored) |

It weakened the "smartphone photo means cancer" shortcut (false alarms on PAD-UFES-20 non-cancers 89% → 63%) and improved the hard distinction, cancer against a benign lookalike (0.72 → 0.78). Its scores are in the table above. Checkpoint 2's first step uses its weights unchanged.

## What was tried to get here: do not repeat

Seven models were trained on the way to checkpoint 1, and two ideas for using it differently became the first step of checkpoint 2. Everything up to checkpoint 3 that was not kept is listed with the reason, so nobody spends time on them again unless something else has changed (see the last section). The ideas after checkpoint 3 (T to AC) are in [IDEAS.md](IDEAS.md).

| # | What was tried | Compared on | Result | Verdict |
| --- | --- | --- | --- | --- |
| 0 | **Baseline.** Every image once per epoch, class-weighted loss | test and outside sets, 3 seeds | Test AUC 0.887; MIDAS 0.60, DDI 0.65. Learned "smartphone photo means cancer": 89% false alarms on PAD-UFES-20 non-cancers | Replaced by A |
| A | **Balanced drawing inside each dataset** | test and outside sets, 3 seeds | Shortcut weakened (89% → 63%), cancers vs lookalikes 0.72 → 0.78. Outside sets unchanged | **Kept: checkpoint 1** |
| B | A + hard lesion types drawn more often (share ∝ square root of count) | validation, 1 seed | Within-source AUC 0.847 against A's 0.869. Fewer false alarms on lookalikes (53% against 64%) but more on everything else | Discarded |
| C | B + random loss of resolution and JPEG quality in training | validation, 1 seed | 0.853 against 0.869. Same trade-off as B | Discarded |
| D | A + Derm7pt (958 atlas photos: 294 cancers, 664 benign) as training-only data | validation, 1 seed | 0.857 against 0.869 | Discarded |
| E | A at 320 × 320 pixels | validation, 1 seed | 0.856 against 0.869; stopped early with no gain | Discarded |
| F | A + Derm7pt at 320 × 320 pixels | validation, 1 seed | 0.862 against 0.869 | Discarded |
| G | Average the three seeds of checkpoint 1 into one prediction | validation | Within-source AUC 0.882 against 0.868 | Kept as part of G+H |
| H | Test-time flips: predict on the image and its three mirror images, average | validation | 0.879 against 0.868 | Kept as part of G+H |
| J | Checkpoint 1's recipe with 552 MIDAS training photos added | validation only (ours and the MIDAS validation part) | MIDAS validation AUC 0.689 → 0.782, but within-source AUC on our validation split 0.888 → 0.876, a supported loss, mostly one seed stopped early at epoch 25 | Discarded at the first step; not scored on test sets. Replaced by O |
| O | J with early stopping off (all seeds run 60 epochs) | test and outside sets, paired comparison with checkpoint 1-TTA | MIDAS test 0.645 → 0.698 and DDI 0.629 → 0.657, both supported. Our test split 0.904 → 0.898, a small supported loss | **Kept: checkpoint 2** (see above for how) |
| I | Checkpoint 2's recipe with 2,907 SCIN phone photos of everyday conditions added as not cancer | validation only (ours and the MIDAS validation part), paired comparison with checkpoint 2 | Our validation split: within-source AUC 0.882 → 0.902, supported. MIDAS validation part: AUC 0.793 → 0.729, a supported loss | Discarded at the first step; not scored on test sets |
| M | Checkpoint 2's recipe with the network at twice the width (11.2 million parameters against 2.8 million) | validation only, paired comparison with checkpoint 2 | Our validation split: within-source AUC 0.882 → 0.885, not supported. MIDAS validation part: AUC 0.793 → 0.773, a supported loss | Discarded at the first step (14 Nov); not scored on test sets |
| N | Checkpoint 2's recipe trained for 120 epochs, not 60 | validation, then test and outside sets, paired comparison with checkpoint 2 | Our test split: AUC 0.898 → 0.930, supported. MIDAS test 0.698 → 0.733, not supported. DDI 0.657 → 0.592, a supported loss | Discarded (14 Nov): the extra training learns our home datasets, not other hospitals |
| L | Self-supervised pretraining (SimCLR, 200 epochs on about 17,000 unlabelled photos), then checkpoint 2's recipe from those weights | validation only, paired comparison with checkpoint 2 | Our validation split: within-source AUC 0.882 → 0.939, supported. MIDAS validation part: AUC 0.793 → 0.753, not supported (against checkpoint 3, 0.818 → 0.753, a supported loss) | Discarded at the first step (14 Nov); not scored on test sets. Pretraining on our own photos learns our home datasets |
| P | Reference only, never a candidate: torchvision ResNet-18 with checkpoint 2's recipe, from ImageNet weights and from random weights | every set, paired comparison | Random start: no better than SkinNet anywhere. ImageNet against random start: home scores +0.05 to +0.07, supported; MIDAS validation −0.029, MIDAS test +0.010, DDI −0.037, none supported. Checkpoint 3 is ahead of both on MIDAS and DDI | Measurement (14–15 Nov): ImageNet weights help at home only. At lr 1e-4 home within-source AUC reaches 0.964 and the outside sets are still below checkpoint 3 (MIDAS test 0.737, DDI 0.631) |
| K | Checkpoint 3's recipe with 5,000 ISIC 2019 dermoscopy images added (2,500 cancers, 2,500 not cancer, one per lesion) | validation only, paired comparison with checkpoint 3 | MIDAS validation part: AUC 0.818 → 0.791, a supported loss. Our validation split: within-source AUC 0.905 → 0.908, flat | Discarded at the first step (15 Nov); not scored on test sets. Dermoscopy does not carry over to clinic photos |
| R | Checkpoint 3's recipe with MILK10k drawn twice as often (27% of the draws → 43%) | validation only, paired comparison with checkpoint 3 | Our validation split: within-source AUC 0.905 → 0.894, a supported loss. MIDAS validation part: 0.818 → 0.815, flat | Discarded at the first step (15 Nov). A new mix of the same photos does not help outside |
| PanDerm | Reference only, never a candidate: PanDerm's skin-pretrained ViT-B/16 (CC BY-NC-ND) on checkpoint 3's data, lr 5e-5, 20 epochs | every set, paired comparison with checkpoint 3 and the ImageNet ResNet-18 | Against checkpoint 3: home within-source AUC +0.06, MIDAS test 0.750 → 0.811, both supported; MIDAS validation +0.016 and DermaCon-IN +0.062, not supported; DDI 0.650 → 0.650. Against ImageNet: level at home, +0.06 to +0.07 on MIDAS, supported | Measurement (16 Nov): skin pretraining helps on MIDAS as well as at home, and not on DDI. Its home scores may be inflated by atlas photos in its pretraining |
| S | Checkpoint 3's recipe with 4,999 biopsied clinical photos from Asan and Hallym (Korea) added, published only as 99 px thumbnails | validation only, paired comparison with checkpoint 3 | Our validation split: within-source AUC 0.905 → 0.924, a supported gain. MIDAS validation part: 0.818 → 0.792, a supported loss | Discarded at the first step (16 Nov). Do not add the thumbnails again; the full-size images are a different question |
| G+H | Both together | test and outside sets, paired comparison | Our test split: AUC 0.890 → 0.904, false alarms 40% → 33%, a supported gain. MIDAS +0.006 and DDI +0.006, neither supported | **Kept: checkpoint 1-TTA** (see above for how) |

The deciding score on validation was the within-source AUC: the AUC inside each dataset, averaged. Seeds of one model differ by about 0.01 on it, so B to F are "no better than A", not "clearly worse".

**What these runs established**

1. **Balancing cancer and not cancer inside each dataset is worth keeping.** It removes the reward for recognising the kind of photo. Every later model should start with it.
2. **Drawing hard lesion types more often is a trade, not a gain.** It cuts false alarms on benign lookalikes by about a fifth and raises them elsewhere. It is only worth revisiting if mistaking moles for cancer becomes the main complaint.
3. **Degrading image quality in training does not help.**
4. **Derm7pt does not help.** One more atlas of moles and melanomas adds little the network had not seen. Its list is still on disk (`data/extra/derm7pt.csv`) and was checked: none of its images duplicates a validation or test image.
5. **Larger images do not help at this network size.** Image detail is not what limits the model.
6. **Averaging seeds and test-time flips are a real but home-only gain.** Together they add about 0.015 AUC on our own test split and cut false alarms there by a sixth, at no training cost. On MIDAS and DDI the change is +0.006, too small to tell from zero. They are now the standard way of predicting (checkpoint 1-TTA), but they do not make a model travel better.
7. **Nothing done with these two data sources moved the outside scores.** The gap between 0.89 at home and about 0.62 outside comes from how different our training images are from other hospitals' photos.
8. **Choosing models on our own validation split cannot find outside gains,** because that split comes from the same two sources as training.
9. **Photos from another hospital in training are the first thing that moved an outside score.** Adding 552 MIDAS photos raised AUC on DDI, a source never seen, by 0.028, and on the MIDAS test part by 0.053, at a cost of 0.007 at home. More sources of training photos is the direction that works.
10. **Early stopping can cut a good run short.** With the learning rate still high the validation score is noisy; one seed stopped at epoch 25 would have improved by 0.03 if left to finish. Train all 60 epochs and keep the best weights.
11. **Easy not-cancer photos help at home and hurt on tumours.** SCIN's everyday rashes raised our own validation score by 0.02 and lowered the MIDAS validation score by 0.06. New data has to contain the hard distinction, cancer against a benign growth, to help outside. SCIN at a much smaller share of the draws is untested.

**Also settled along the way** (reasons are in [DECISIONS.md](../DECISIONS.md))

- A five-class version (BCC / SCC / melanoma / benign / other) was set up and dropped before any result, in favour of cancer / not cancer as stage one.
- Skin-tone analysis is left for a later stage.
- Training on the CPU takes about 7 minutes per epoch; on the GPU, 18 seconds. Always train on the GPU.

## How an idea is judged now, and what is next

Each idea has a letter and changes one thing in the current checkpoint. Its rule is written in [DECISIONS.md](../DECISIONS.md) before it trains (section 26, as applied in section 32):

1. **Step 1, validation sets only.** Three seeds, paired with the checkpoint on `val` (within-source AUC), `midas_val` and `ddi_val`. It passes if it is higher on all three and lower on none by a supported amount.
2. **Step 2, test sets, scored once.** An idea that passed is scored on every test set, each look recorded in `runs/final_ledger.csv`.
3. **A candidate** if nothing is lower by a supported amount and at least one outside set is higher by a supported amount. Whether it becomes the next checkpoint is the user's decision.

A discarded idea's result goes into [IDEAS.md](IDEAS.md) and is reported to the user; once the user has seen it, its config, runs and weights are deleted (DECISIONS.md, section 36).

**Next (26 Nov 2026), each started only on the user's word:**
- **Z, MixStyle** and **AA, mixup and CutMix** (DECISIONS.md, section 32): written against checkpoint 4 and judged against it as written. Z's seed 0 is done and seed 1 stopped at epoch 27 of 60.
- **SkinNet-KD:** SkinNet trained from random weights to follow PanDerm's predictions. A separately named model, never one of SkinNet's checkpoints (section 33).
- **Settled:** more epochs. Checkpoint 2 (idea N) and checkpoint 5 (idea AC) at 120 epochs gain at home and not outside (section 35).

**Later stages, not ideas for this one:** which type of cancer it is, and whether darker skin needs its own model or one combined model.

**A limit to keep in mind.** The largest known lever for performance on new hospitals is starting from pretrained weights, which this track rules out. A from-scratch network of this size on this much data stays well below the pretrained model: PanDerm is ahead of checkpoint 5 on every outside set, by 0.05 to 0.21 AUC, and by a supported amount on all but DermaCon-IN and MCR-SL ([README](../README.md)). An ImageNet ResNet-18 (idea P, 14–15 Nov) was far ahead at home and behind checkpoint 3 on MIDAS and DDI.

**When a discarded idea is worth another look:** if the training data changes a lot (new sources) or the network changes a lot (much larger), results B to H no longer apply as they stand, because they were all measured with this data and this network.
