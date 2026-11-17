# Checkpoints

The full list of ideas by letter, kept and discarded, is in [IDEAS.md](IDEAS.md).

A checkpoint is a model we have decided to keep and build on. Each new idea starts from the latest checkpoint and has to beat it to become the next one. A checkpoint also records everything that was tried on the way to it, so the same experiments are not run twice.

Scores are the mean of three seeds. Details and confidence intervals are in [RESULTS.md](RESULTS.md).

## Names

| Model | Name in `results/` and `runs/` | Built from |
| --- | --- | --- |
| Baseline | `skinnet` | `configs/skinnet.yaml` |
| Checkpoint 1 | `skinnet-c1` | `configs/skinnet-c1.yaml` |
| Checkpoint 2 | `skinnet-c2` | `skinnet-c1`'s three seeds, predicting with test-time flips, and their average |
| Checkpoint 3 | `skinnet-c3` | `configs/skinnet-c3.yaml` |
| Checkpoint 4 | `skinnet-c4` | `configs/skinnet-c4.yaml` (was idea Q, `skinnet-idea-q`) |
| PanDerm checkpoint 1 | `panderm-c` | `configs/panderm-c.yaml` (PanDerm track, idea PC; DECISIONS.md, section 24) |

One name per model, the same for its config, run folder and result folder. A run folder `runs/<name>/` holds the seeds (`s0`, `s1`, `s2`) and their average (`avg`); `results/<name>/metrics.csv` is the model's full score table. An idea being tested is `skinnet-idea-<letter>`; if kept it is renamed to the next checkpoint, `skinnet-c5`, and if discarded it is deleted. The rule is in [DECISIONS.md](../DECISIONS.md), section 15.

## PanDerm checkpoint 1 (17 Nov 2026): the PanDerm track's current model

Since 17 Nov there are two tracks (DECISIONS.md, section 24): the from-scratch one, whose current model is checkpoint 4 below, and the PanDerm one, which starts from PanDerm's pretrained weights (Yan et al., Nature Medicine 2025; CC BY-NC-ND 4.0, non-commercial, not redistributed). One of the two is picked at the end.

PanDerm's ViT-L/16, the network of its paper, fine-tuned on checkpoint 4's training data with our recipe (idea PA): the head trained alone for 3 epochs, then every layer with a learning rate that shrinks by 0.86 per block towards the input; early stopping 4 epochs after the best.

| | |
| --- | --- |
| Training data | The 18,022 images of checkpoint 4 |
| Training | 3 epochs head only, then up to 20; about 10 minutes an epoch with gradient checkpointing, about 2 hours a seed |
| Config | `configs/panderm-c.yaml` (weights `data/weights/panderm_large.pth`, local) |
| Run folder | `runs/panderm-c`: seeds in `s0`, `s1`, `s2`, their average in `avg` |
| Weights | `checkpoints/panderm-1/` (local, git-ignored) |
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

## Checkpoint 4 (14 Nov 2026): the from-scratch track's current model

Checkpoint 3's recipe with 4,908 biopsy-confirmed clinical close-ups from MILK10k added to training (idea Q): cancers and the benign growths that resemble them, from five centres not otherwise in the data (Vienna, Ankara, Queensland, Memorial Sloan Kettering, Skopje). Predicts the same way (test-time flips, three-seed average).

| | |
| --- | --- |
| Training data | 18,022 images: the 13,114 of checkpoint 3 plus 4,908 from MILK10k (3,620 cancer, 1,288 not cancer; training only) |
| Training | 60 epochs, early stopping off, about 30 minutes per seed |
| Config | `configs/skinnet-c4.yaml` |
| Run folder | `runs/skinnet-c4`: seeds in `s0`, `s1`, `s2`, their average in `avg` |
| Weights | `checkpoints/checkpoint-4/` (local, git-ignored) |
| Git tag | `checkpoint-4` |

| Score | Checkpoint 3 | Checkpoint 4 | Paired difference [95% interval] |
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

**From now on** every new idea is compared with checkpoint 4. Idea L, specified against checkpoint 3 before checkpoint 4 existed, was judged against checkpoint 3 as written, and failed (DECISIONS.md, section 17).

## Checkpoint 3 (13 Nov 2026)

Checkpoint 1's recipe trained again with photos from a second kind of source: 552 biopsy-confirmed hospital photos from MIDAS. Predicts the same way as checkpoint 2 (test-time flips, three-seed average).

| | |
| --- | --- |
| Training data | 13,114 images: the 12,562 of checkpoint 1 plus 552 from MIDAS (`midas_train`) |
| Training | 60 epochs, early stopping off, about 20 minutes per seed |
| Config | `configs/skinnet-c3.yaml` |
| Run folder | `runs/skinnet-c3`: seeds in `s0`, `s1`, `s2`, their average in `avg` |
| Weights | `checkpoints/checkpoint-3/` (local, git-ignored) |
| Git tag | `checkpoint-3` |

| Score | Checkpoint 2 | Checkpoint 3 | Paired difference [95% interval] |
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

**From now on** every new idea is compared with checkpoint 3 after the same treatment (three seeds, 60 epochs, test-time flips, seed average), and chosen on the MIDAS validation part as well as our own.

## Checkpoint 2 (12 Nov 2026)

Checkpoint 1's three trained models, used better when predicting. Nothing was retrained.

| | |
| --- | --- |
| What it is | For each image, each of the three seeds predicts on the image and its three mirror images (left-right, up-down, both); the twelve cancer probabilities are averaged into one |
| Weights | The same as checkpoint 1; copies with the prediction settings are in `checkpoints/checkpoint-2/` (local, git-ignored) |
| Run folder | `runs/skinnet-c2`: seeds in `s0`, `s1`, `s2`, their average in `avg` |
| Git tag | `checkpoint-2` |
| Built with | `python -m skincancer.derive` (commands below) |

| Score | Baseline | Checkpoint 1 | Checkpoint 2 |
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

**How it became a checkpoint.** The rule written before the test (DECISIONS.md, section 10) asked for a supported gain on MIDAS or DDI, and by that rule it was first discarded. On seeing the result the team decided that a free, supported gain at home with nothing worse anywhere counts as working, and restored it. That was a change of rule after the result, recorded here so the outside numbers are not read as better than they are.

**Until checkpoint 3** every new idea was compared with checkpoint 2, and got the same treatment before the comparison: three seeds, test-time flips, seed average.

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
| Weights | `checkpoints/checkpoint-1/s{0,1,2}/best.pt` (local, git-ignored) |

It weakened the "smartphone photo means cancer" shortcut (false alarms on PAD-UFES-20 non-cancers 89% → 63%) and improved the hard distinction, cancer against a benign lookalike (0.72 → 0.78). Its scores are in the table above. Checkpoint 2 uses its weights unchanged.

## What was tried to get here: do not repeat

Seven models were trained on the way to checkpoint 1, and two ideas for using it differently led to checkpoint 2. Everything that was not kept is listed with the reason, so nobody spends time on them again unless something else has changed (see the last section).

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
| O | J with early stopping off (all seeds run 60 epochs) | test and outside sets, paired comparison with checkpoint 2 | MIDAS test 0.645 → 0.698 and DDI 0.629 → 0.657, both supported. Our test split 0.904 → 0.898, a small supported loss | **Kept: checkpoint 3** (see above for how) |
| I | Checkpoint 3's recipe with 2,907 SCIN phone photos of everyday conditions added as not cancer | validation only (ours and the MIDAS validation part), paired comparison with checkpoint 3 | Our validation split: within-source AUC 0.882 → 0.902, supported. MIDAS validation part: AUC 0.793 → 0.729, a supported loss | Discarded at the first step; not scored on test sets |
| M | Checkpoint 3's recipe with the network at twice the width (11.2 million parameters against 2.8 million) | validation only, paired comparison with checkpoint 3 | Our validation split: within-source AUC 0.882 → 0.885, not supported. MIDAS validation part: AUC 0.793 → 0.773, a supported loss | Discarded at the first step (14 Nov); not scored on test sets |
| N | Checkpoint 3's recipe trained for 120 epochs, not 60 | validation, then test and outside sets, paired comparison with checkpoint 3 | Our test split: AUC 0.898 → 0.930, supported. MIDAS test 0.698 → 0.733, not supported. DDI 0.657 → 0.592, a supported loss | Discarded (14 Nov): the extra training learns our home datasets, not other hospitals |
| L | Self-supervised pretraining (SimCLR, 200 epochs on about 17,000 unlabelled photos), then checkpoint 3's recipe from those weights | validation only, paired comparison with checkpoint 3 | Our validation split: within-source AUC 0.882 → 0.939, supported. MIDAS validation part: AUC 0.793 → 0.753, not supported (against checkpoint 4, 0.818 → 0.753, a supported loss) | Discarded at the first step (14 Nov); not scored on test sets. Pretraining on our own photos learns our home datasets |
| P | Reference only, never a candidate: torchvision ResNet-18 with checkpoint 3's recipe, from ImageNet weights and from random weights | every set, paired comparison | Random start: no better than SkinNet anywhere. ImageNet against random start: home scores +0.05 to +0.07, supported; MIDAS validation −0.029, MIDAS test +0.010, DDI −0.037, none supported. Checkpoint 4 is ahead of both on MIDAS and DDI | Measurement (14–15 Nov): ImageNet weights help at home only. At lr 1e-4 home within-source AUC reaches 0.964 and the outside sets are still below checkpoint 4 (MIDAS test 0.737, DDI 0.631) |
| K | Checkpoint 4's recipe with 5,000 ISIC 2019 dermoscopy images added (2,500 cancers, 2,500 not cancer, one per lesion) | validation only, paired comparison with checkpoint 4 | MIDAS validation part: AUC 0.818 → 0.791, a supported loss. Our validation split: within-source AUC 0.905 → 0.908, flat | Discarded at the first step (15 Nov); not scored on test sets. Dermoscopy does not carry over to clinic photos |
| R | Checkpoint 4's recipe with MILK10k drawn twice as often (27% of the draws → 43%) | validation only, paired comparison with checkpoint 4 | Our validation split: within-source AUC 0.905 → 0.894, a supported loss. MIDAS validation part: 0.818 → 0.815, flat | Discarded at the first step (15 Nov). A new mix of the same photos does not help outside |
| PanDerm | Reference only, never a candidate: PanDerm's skin-pretrained ViT-B/16 (CC BY-NC-ND) on checkpoint 4's data, lr 5e-5, 20 epochs | every set, paired comparison with checkpoint 4 and the ImageNet ResNet-18 | Against checkpoint 4: home within-source AUC +0.06, MIDAS test 0.750 → 0.811, both supported; MIDAS validation +0.016 and DermaCon-IN +0.062, not supported; DDI 0.650 → 0.650. Against ImageNet: level at home, +0.06 to +0.07 on MIDAS, supported | Measurement (16 Nov): skin pretraining helps on MIDAS as well as at home, and not on DDI. Its home scores may be inflated by atlas photos in its pretraining |
| S | Checkpoint 4's recipe with 4,999 biopsied clinical photos from Asan and Hallym (Korea) added, published only as 99 px thumbnails | validation only, paired comparison with checkpoint 4 | Our validation split: within-source AUC 0.905 → 0.924, a supported gain. MIDAS validation part: 0.818 → 0.792, a supported loss | Discarded at the first step (16 Nov). Do not add the thumbnails again; the full-size images are a different question |
| G+H | Both together | test and outside sets, paired comparison | Our test split: AUC 0.890 → 0.904, false alarms 40% → 33%, a supported gain. MIDAS +0.006 and DDI +0.006, neither supported | **Kept: checkpoint 2** (see above for how) |

The deciding score on validation was the within-source AUC: the AUC inside each dataset, averaged. Seeds of one model differ by about 0.01 on it, so B to F are "no better than A", not "clearly worse".

**What these runs established**

1. **Balancing cancer and not cancer inside each dataset is worth keeping.** It removes the reward for recognising the kind of photo. Every later model should start with it.
2. **Drawing hard lesion types more often is a trade, not a gain.** It cuts false alarms on benign lookalikes by about a fifth and raises them elsewhere. It is only worth revisiting if mistaking moles for cancer becomes the main complaint.
3. **Degrading image quality in training does not help.**
4. **Derm7pt does not help.** One more atlas of moles and melanomas adds little the network had not seen. Its list is still on disk (`data/extra/derm7pt.csv`) and was checked: none of its images duplicates a validation or test image.
5. **Larger images do not help at this network size.** Image detail is not what limits the model.
6. **Averaging seeds and test-time flips are a real but home-only gain.** Together they add about 0.015 AUC on our own test split and cut false alarms there by a sixth, at no training cost. On MIDAS and DDI the change is +0.006, too small to tell from zero. They are now the standard way of predicting (checkpoint 2), but they do not make a model travel better.
7. **Nothing done with these two data sources moved the outside scores.** The gap between 0.89 at home and about 0.62 outside comes from how different our training images are from other hospitals' photos.
8. **Choosing models on our own validation split cannot find outside gains,** because that split comes from the same two sources as training.
9. **Photos from another hospital in training are the first thing that moved an outside score.** Adding 552 MIDAS photos raised AUC on DDI, a source never seen, by 0.028, and on the MIDAS test part by 0.053, at a cost of 0.007 at home. More sources of training photos is the direction that works.
10. **Early stopping can cut a good run short.** With the learning rate still high the validation score is noisy; one seed stopped at epoch 25 would have improved by 0.03 if left to finish. Train all 60 epochs and keep the best weights.
11. **Easy not-cancer photos help at home and hurt on tumours.** SCIN's everyday rashes raised our own validation score by 0.02 and lowered the MIDAS validation score by 0.06. New data has to contain the hard distinction, cancer against a benign growth, to help outside. SCIN at a much smaller share of the draws is untested.

**Also settled along the way** (reasons are in [DECISIONS.md](../DECISIONS.md))

- A five-class version (BCC / SCC / melanoma / benign / other) was set up and dropped before any result, in favour of cancer / not cancer as stage one.
- Skin-tone analysis is left for a later stage.
- Training on the CPU takes about 7 minutes per epoch; on the GPU, 18 seconds. Always train on the GPU.

## Ideas not tried yet

Each idea has a letter, continuing from A to H above. The rule is the same for every one:

- **If it works, it becomes the next checkpoint.** "Works" means a gain over the current checkpoint that the paired comparison supports, with nothing clearly worse elsewhere. Whether the gain is at home or on the outside sets is stated with it; a gain on the outside sets is the one we are after.
- **If it does not, it is discarded:** its config and trained models are deleted and it moves to the "do not repeat" table with its result and the reason.
- One idea at a time where possible, so a gain or a loss can be traced to one change.

**Setup step: done (13 Nov 2026).** MIDAS is split by patient into a training part (552 images), an outside validation part (414, never trained on, used to choose models) and a test part (414). DDI is untouched. Ideas are now chosen on the outside validation part as well as our own.

Every idea on the first list has been run (16 Nov 2026): Q is checkpoint 4, P is a reference, the rest are in the table above, S included. The PanDerm reference (DECISIONS.md, section 23) is done too. On 17 Nov the team made PanDerm a second track beside this one (section 24); its ideas are listed in [IDEAS.md](IDEAS.md). The from-scratch track stays at checkpoint 4. Data details are in [DATA_PLAN.md](DATA_PLAN.md).

**Order (16 Nov 2026):** N, M, L, K, R and S are done, all discarded, and Q became checkpoint 4 (table above). P (the ImageNet reference) is done, its lower learning rate included (table above). The PanDerm reference is done (table above). On the PanDerm track, idea PC is PanDerm checkpoint 1 (section 24; above). From Q on, an idea must be higher on every set (section 19). Reasons are in [DECISIONS.md](../DECISIONS.md), sections 16 and 19.

**Later stages, not ideas for this one:** which type of cancer it is, and whether darker skin needs its own model or one combined model.

**A limit to keep in mind.** The largest known lever for performance on new hospitals is starting from pretrained weights, which this project has ruled out. A from-scratch network of this size on this much data is likely to stay well below what pretrained models reach on DDI and MIDAS, whatever comes of I to N. (Idea P, 14–15 Nov: with our training data it did not. An ImageNet ResNet-18, at either learning rate, is far ahead at home and behind checkpoint 4 on MIDAS and DDI.)

**When a discarded idea is worth another look:** if the training data changes a lot (new sources) or the network changes a lot (much larger), results B to H no longer apply as they stand, because they were all measured with this data and this network.

## How to use a checkpoint

```bash
# checkpoint 1, one seed (weights are local, not in git)
python -m skincancer.predict --run checkpoints/checkpoint-1/s0 --splits val

# checkpoint 2: rebuild it from checkpoint 1's runs
python -m skincancer.derive tta --exp skinnet-c1 --name skinnet-c2
for s in 0 1 2; do python -m skincancer.predict --run runs/skinnet-c2/s$s --splits val test; done
python -m skincancer.derive ensemble --exp skinnet-c2 --splits val test
python -m skincancer.compare --a skinnet-c1 --b skinnet-c2 --splits val test

# checkpoint 3: train it (flips are in its config), then average the seeds
for s in 0 1 2; do python -m skincancer.train --config configs/skinnet-c3.yaml --seed $s; done
python -m skincancer.derive ensemble --exp skinnet-c3 --splits val test midas_val

# return the code to the state that produced a checkpoint
git checkout checkpoint-3
```

The weights are not in the repository (`checkpoints/` is git-ignored). Retraining from the tagged code with seeds 0, 1 and 2 reproduces them.
