# Decisions

The rules this project runs by. The code enforces them. Changing a rule means changing it here, in the code, and re-running whatever it touches.

## 1. Fresh start (12 Nov 2026)

The project was restarted on 12 Nov 2026. The classifier is now one convolutional network of our own (SkinNet), trained from random weights. Nothing in it is pretrained and no layer is frozen.

- Everything from before the restart was removed from the working tree: the pretrained-backbone fine-tuning runs, image-processing ablation, Indian-prior loss, texture fusion, Deep CORAL, Derm Foundation heads, model ensembles, the "not a tumour" gates, and their results, figures and write-ups.
- That work is kept in git under the tag `pre-fresh-start` (`git checkout pre-fresh-start`). Its local run folders, results and figures were moved to `archive/pre-fresh-start/`, which git ignores.
- The label map and the near-duplicate groups are the same files as before. The labels became cancer / not cancer and the train / validation / test splits were rebuilt the same day (sections 2 and 3). The three outside datasets used for testing (DDI, MIDAS, DermaCon-IN) hold the same images and the same cancer labels as before.
- The ledger of test-set looks starts empty. The test sets themselves are not new: the earlier experiments were scored on them, and those scores were seen. No choice in this file was made from them.

## 2. Classes and labels

**Stage one is one question: is this image a skin cancer or not?** Two classes, `cancer` and `not_cancer` (decided 12 Nov 2026).

Not part of stage one, by decision: which type of cancer it is, and how results differ between lighter and darker skin. Whether darker skin needs its own model or one combined model serves everyone is the next stage's question. Skin type stays recorded in the manifest for that, but it plays no part in the labels, the splits or the reported results now.

| Class | What is in it | Images in train / val / test |
| --- | --- | --- |
| `cancer` | Basal cell carcinoma, squamous cell carcinoma (with Bowen's disease), melanoma, and rare skin cancers (Kaposi sarcoma, mycosis fungoides, DFSP, Merkel cell and others) | 3,176 |
| `not_cancer` | Skin conditions that are not tumours (psoriasis, eczema, acne, scabies, lupus and about a hundred more); benign tumours that resemble cancer (nevi, seborrheic keratosis, dermatofibroma, solar lentigo, pyogenic granuloma); other benign growths and birthmarks (cysts, syringoma, epidermal nevi, port wine stains) | 14,776 |

- **Actinic keratosis is in neither class.** It is precancerous, so calling it either would be wrong. It is never trained on. AK images form the `ak_side` set, where we report how many the model calls cancer.
- **Diagnosis mapping.** Ordered rules in `src/skincancer/labels.py` map each dataset's diagnosis strings to a tumour label (`label_4class`). `data/label_map.csv` holds every (dataset, diagnosis) pair, was reviewed row by row ([docs/LABEL_MAP_REVIEW.md](docs/LABEL_MAP_REVIEW.md)), and is the source of truth. `build_manifest` turns it into `label_cancer`, the column the model is trained on, and `label_fine`, a finer description (BCC, melanoma, non-tumour and so on) used only to break results down.
- **Where "not a tumour" comes from.** Fitzpatrick17k marks each of its diagnoses malignant, benign or non-neoplastic; that marking decides its `not_cancer` images. DDI uses its biopsy result for every image. DermaCon-IN's everyday clinic conditions count as not cancer.
- **Excluded:** 17 Fitzpatrick17k images its own quality check marked as wrongly labelled, 2 DermaCon-IN images of insufficient quality, 593 MIDAS images whose diagnosis does not say whether they are cancer ("benign-other", "malignant-other", controls), and AK images of patients who also have another lesion in the data.
- Images with no recorded skin type are kept (915 of them in train / val / test), since stage one does not use skin type.

## 3. Splits

Frozen in `data/manifest.csv` (`split` column) and `data/splits/`. Rebuilt on 12 Nov 2026 for the two-class design; nobody re-splits after this.

| Split | Images | Cancer | What it is |
| --- | --- | --- | --- |
| `train` | 12,562 | 2,220 | Fitzpatrick17k + PAD-UFES-20 |
| `val` | 2,695 | 478 | same sources |
| `test` | 2,695 | 478 | same sources |
| `ddi_test` | 656 | 171 | DDI (Stanford, biopsy-confirmed) |
| `midas_test` | 1,380 | 802 | MRA-MIDAS clinical photos (Stanford, biopsy-confirmed), tumours only |
| `dermacon_test` | 65 | 19 | DermaCon-IN (India), tumours |
| `dermacon_other` | 5,379 | 0 | DermaCon-IN, everyday conditions |
| `ak_side` | 703 | none labelled | Actinic keratoses |

- Train / val / test is a 70 / 15 / 15 split, stratified by type of lesion and by dataset, so every split has the same mix (17.7% cancer in each).
- **No leakage.** Near-duplicate images (perceptual hash distance ≤ 2, or embedding cosine ≥ 0.90) and images of the same patient form a group. A group never spans two splits; `make_splits` asserts this. The duplicate search used an ImageNet EfficientNet-B0 as a similarity measure. That is data preparation, and it is the only place pretrained weights appear in this repository.
- **The three outside test datasets (DDI, MIDAS, DermaCon-IN) hold the same images and cancer labels as before the restart.** The rebuilt manifest was checked against the previous one: paths, diagnoses, tumour labels, skin types and groups are identical for all 27,081 rows.
- **DDI images never appear in a figure** (licence terms). The repository stays private because the manifest carries DDI metadata.

## 4. What trains, what selects, what tests

- **Training data:** `train`.
- **Selection:** `val` picks the checkpoint and stops training. Test splits never choose anything.
- **Main result:** `test`, overall, per dataset, and per type of lesion (which cancers are missed, which harmless conditions raise false alarms).
- **Test-only sets** (`ddi_test`, `dermacon_test`, `dermacon_other`, `midas_test`) are never scored during training. `predict --final` scores them once per finished run and records it in `runs/final_ledger.csv`; a second look is refused.
- **Known limits of this design, stated before any run:**
  - **The two sources differ in how much cancer they hold.** 70% of PAD-UFES-20 images are cancer against 13% of Fitzpatrick17k images, and PAD-UFES-20 images are smartphone photos while Fitzpatrick17k images are atlas photos. The network can score well by recognising the kind of photo instead of the lesion. `test` is therefore reported per dataset, and the outside sets (different hospitals, cameras and countries) are the real check.
  - **Most `not_cancer` images are easy.** 82% of them are rashes and other non-tumour conditions that look little like a tumour. A high overall score can hide a weak result on the hard part, telling a cancer from a benign tumour that resembles it. The per-type breakdown and MIDAS (tumours only) show that part.
  - The classes are unbalanced: 18% of `train` is cancer. The loss is class-balanced (section 6).

## 5. The network: SkinNet

Defined in `src/skincancer/models.py`. A residual CNN, small enough to train on a laptop CPU.

| Part | Layers | Output for a 224 × 224 input |
| --- | --- | --- |
| Stem | 3 × (3×3 conv, batch norm, ReLU), widths 16, 16, 32, the first with stride 2; then 3×3 max-pool, stride 2 | 32 × 56 × 56 |
| Stage 1 | 2 residual blocks, 32 channels | 32 × 56 × 56 |
| Stage 2 | 2 residual blocks, 64 channels, first one stride 2 | 64 × 28 × 28 |
| Stage 3 | 2 residual blocks, 128 channels, first one stride 2 | 128 × 14 × 14 |
| Stage 4 | 2 residual blocks, 256 channels, first one stride 2 | 256 × 7 × 7 |
| Head | global average pool, dropout 0.3, linear to 2 classes | 2 |

- A residual block is two 3×3 convolutions, each followed by batch norm, with a skip connection around them. Where a block changes resolution or width, the skip is average-pool then 1×1 convolution.
- 2.8 million parameters, 17 convolution layers on the main path.
- Weights start random (Kaiming normal). The last batch norm of each block starts at zero, so every block begins as the identity. Each seed gives a different start.

## 6. Training

Fixed on 12 Nov 2026, before any SkinNet run was scored on a test set. The config is `configs/skinnet.yaml`.

| Setting | Value |
| --- | --- |
| Layers trained | all, from the first step |
| Optimiser | AdamW, learning rate 1e-3, weight decay 0.05 |
| Schedule | 3 epochs linear warm-up, then cosine decay to zero at epoch 60 |
| Stopping | best `val` AUC is kept; stop after 15 epochs without improvement |
| Loss | cross-entropy, label smoothing 0.1, class-balanced weights |
| Batch | 32 images, 224 × 224 |
| Augmentation | random resized crop (40–100% of the image), horizontal and vertical flips, 90° turns, rotation ±30°, brightness and contrast ±20%, light blur, 1–3 erased patches |
| Not used | hue or saturation changes: they change apparent skin tone |
| Seeds | 0, 1, 2 |

- Class-balanced weights give each class the same total weight in the loss, so the 10,342 `not_cancer` images do not outvote the 2,220 cancers.
- Evaluation images are resized to 256 on the shorter side and centre-cropped to 224.
- One epoch is 12,562 images, about 6 minutes on the development laptop (no GPU), so a seed takes up to 6 hours.
- **Expectation, stated in advance:** the score on `test` will look good, mostly because rashes are easy to tell from cancers and because of the photo-type shortcut above. Expect clearly lower scores on DDI, MIDAS and DermaCon-IN; those are the numbers to judge the model by.

## 7. Metrics and statistics

- **AUC** for cancer vs not cancer, on every set that holds both.
- **Missed cancers and false alarms at a fixed operating point ("op90"):** per seed, the threshold on P(cancer) that catches 90% of cancers in `val`, applied unchanged to every other set. Missed cancers are the share of cancers below the threshold; false alarms the share of non-cancers at or above it. The same two rates at the plain 0.5 threshold are also reported.
- **Per type of lesion** on `test`, MIDAS and DermaCon-IN: the missed-cancer rate for each cancer type, and the false-alarm rate for each kind of non-cancer.
- On `ak_side` and `dermacon_other`, the share of images called cancer.
- Every number is the mean of the three seeds, with a 95% confidence interval from 1,000 bootstrap resamples of groups (not single images). Subgroups under 10 images are not reported.

## 8. Second round: training that the shortcut cannot exploit (12 Nov 2026)

Written after the baseline results ([docs/RESULTS.md](docs/RESULTS.md)) and before any of these models was trained. No new data is used.

**What the baseline showed.** It learned that a smartphone photo usually means cancer (89% false alarms on PAD-UFES-20 non-cancers), and it could not tell a cancer from a benign tumour that resembles it (81% false alarms on those).

**Three variants**, each changing only how training images are drawn and augmented. The network, optimiser, schedule and data are the baseline's.

| Variant | Config | Change |
| --- | --- | --- |
| A | `skinnet-c1` | Inside each dataset, cancer and not cancer are drawn equally often, so the kind of photo no longer predicts the class |
| B | `skinnet-c1type` | As A, and inside each dataset × class a lesion type's share of the draws is proportional to the square root of its image count. Benign lookalikes go from 6% to 11% of the images drawn, rare cancers from 7% to 9% |
| C | `skinnet-c1type_q` | As B, plus random loss of resolution and JPEG quality during training, so image quality says less about the source |

- Balanced drawing replaces the class-weighted loss (the loss is unweighted for A, B and C). An epoch still draws 12,562 images, now with replacement.
- **A new selection score: within-source AUC.** The AUC is computed inside each dataset and the two are averaged. Telling the datasets apart cannot raise it. A, B and C pick their checkpoint on it. The baseline picked on plain AUC, so its within-source AUC on `val` is reported next to theirs for reference.

**How one is chosen.**
1. Train seed 0 of A, B and C.
2. Compare them on `val` only: within-source AUC first; the AUC on cancers against benign lookalikes (`hard` subset) breaks a tie of less than 0.005.
3. Train seeds 1 and 2 of the winner only, and score that one model on `test` and the outside sets, once.

The other two variants are never scored on `test` or the outside sets. Training writes their `test` predictions to disk like every run, but those files are not evaluated.

**What would count as helping:** on the outside sets, a higher AUC on MIDAS and DDI than the baseline's 0.60 and 0.65; on our own `test` split, fewer false alarms on PAD-UFES-20 non-cancers and on benign lookalikes at the same share of cancers caught. Plain AUC on `test` may fall, since part of the baseline's 0.887 came from the shortcut.

**If none helps,** the next steps are in [docs/DATA_PLAN.md](docs/DATA_PLAN.md) (Derm7pt first) and then larger images.

**Outcome (12 Nov 2026):** A was chosen and became checkpoint 1 ([docs/CHECKPOINTS.md](docs/CHECKPOINTS.md)). B and C were discarded: their configs, trained models and the two code options only they used were removed. The comparison is kept in [docs/RESULTS.md](docs/RESULTS.md).

## 9. Third round: more hard cases and larger images (12 Nov 2026)

Written after the second-round results and before any of these models was trained.

**What the second round showed.** Balanced drawing (variant A) weakened the shortcut and improved the hard distinction on our own test split, but the outside sets did not move (MIDAS 0.63, DDI 0.62). Variant A's training setup is the starting point for everything below.

**Derm7pt as training-only data.** 958 clinical photos from an Italian atlas with a usable label: 294 cancers (252 melanomas, 4 of them skin metastases, and 42 BCCs) and 664 benign tumours, mostly nevi. The duplicate search found none of them among the 13,573 validation and test images. It has no SCC and records no skin type.
- It is kept outside the manifest (`data/extra/derm7pt.csv`, built by `python -m skincancer.data.extra derm7pt`) and joins `train` only. `val`, `test` and the outside sets are untouched, so models with and without it are scored on exactly the same images.
- Any Derm7pt image that near-duplicates a validation or test image is dropped, with the same thresholds as the main duplicate search.
- In the balanced drawing it counts as a third dataset: cancer and not cancer are drawn equally often inside it.
- Derm7pt images never appear in a figure (licence terms), and its list is not committed.

**Three variants.**

| Variant | Config | Change from variant A |
| --- | --- | --- |
| D | `skinnet_d7` | Derm7pt added to training |
| E | `skinnet_320` | Images at 320 × 320 pixels instead of 224 × 224 (evaluation: shorter side to 366, centre crop 320) |
| F | `skinnet_d7_320` | Both |

**How one is chosen:** the rule of section 8. Seed 0 of D, E and F is compared with seed 0 of variant A (checkpoint 1) on `val`: within-source AUC first, and the AUC on cancers against benign lookalikes breaks a tie of less than 0.005. If none beats checkpoint 1 by 0.005 or more on within-source AUC, checkpoint 1 stays the current model and nothing new is scored outside. Otherwise the winner gets seeds 1 and 2, is scored once on `test` and the outside sets, and becomes the next checkpoint.

**What would count as helping:** a higher AUC on MIDAS and DDI than variant A's 0.63 and 0.62. An improvement on our own test split alone would repeat the second round's outcome.

**Outcome (12 Nov 2026):** none of D, E and F beat checkpoint 1 on `val` (0.857, 0.856 and 0.862 against 0.869), so checkpoint 1 stays the current model and nothing new was scored on `test` or the outside sets. A full list of what has been tried, so it is not repeated, is in [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md). The three variant configs and their trained models were removed. The code for training-only extra data (`data/extra.py`, `data.extra_train`) is kept for the next data steps; no config uses it now.

## 10. Ideas G and H: averaging seeds and test-time flips (12 Nov 2026)

Written before either was computed. Both change only how checkpoint 1's three trained models are used to predict; nothing is retrained.

- **G, seed average** (`skinnet-c1_ens`): the three seeds' cancer probabilities are averaged into one prediction per image.
- **H, test-time flips** (the seeds in `runs/skinnet-c2`): each seed predicts on the image and its three mirror images (left-right, up-down, both) and averages the four.
- **G + H** (`skinnet-c2`): the seed average of H's predictions.

**Step 1, validation only.** Within-source AUC on `val` for all three, against checkpoint 1's 0.868 (mean of its seeds). A candidate passes with a gain of 0.005 or more. If none passes, G and H are discarded and nothing is scored outside.

**Step 2, one candidate.** The passing candidate with the highest validation score is scored once on `test` and the outside sets. Averaging existing outside predictions is still a new look at those sets, so it is recorded in the ledger like any other.

**Step 3, does it become a checkpoint?** Differences this small cannot be read from two separate confidence intervals, so the comparison is paired: the same resampled images are scored by both models and the difference in AUC gets its own 95% interval (`python -m skincancer.compare`). The candidate becomes checkpoint 2 if that interval is above zero on MIDAS or on DDI and is not below zero on the other. Otherwise it is discarded, even if it gained on our own test split.

**Outcome (12 Nov 2026):** all three passed step 1 (within-source AUC on `val` 0.882, 0.879 and 0.888 against 0.868). G + H was scored outside. Its paired gain in AUC was +0.006 [−0.001, 0.014] on MIDAS and +0.006 [−0.004, 0.016] on DDI. Both intervals include zero, so it was discarded as a checkpoint, although it gained +0.015 on our own test split. The paired comparison tool (`skincancer.compare`) is kept, because every later idea is judged the same way. The tool that built G and H was removed with them.

**Amended the same day:** G + H was restored and is checkpoint 2. The rule above was written before the test and G + H did not meet it; the decision to keep it anyway was made after seeing the result, on the grounds that the gain on our own test split is supported, costs nothing, and nothing is worse anywhere. This is recorded as a change of rule, not as G + H having passed. From here on an idea "works" if the paired comparison supports a gain over the current checkpoint and nothing is clearly worse elsewhere; where the gain is (at home or outside) is always stated with it. The tool that builds G + H (`skincancer.derive`) is back in the repository.

## 11. MIDAS split three ways; idea J (12 Nov 2026)

Written before the split was made and before idea J was trained.

**Why.** Every model so far was chosen on a validation split drawn from the same two sources as training, and no choice made that way moved the outside scores. Two things were missing: outside photos to choose models on, and outside-like photos to train on. MIDAS can supply both, at the price of a smaller MIDAS test set.

**The split.** MIDAS's 1,380 biopsy-confirmed tumour photos are split by patient, stratified by type of lesion:

| Part | Share | Use |
| --- | --- | --- |
| `midas_train` | 40% | May be added to training (idea J). Unused unless an idea says so |
| `midas_val` | 30% | Outside validation: never trained on, used to choose between models |
| `midas_test` | 30% | Test only, scored once per model as before |

- This cannot be undone: once models are chosen on `midas_val` or trained on `midas_train`, those images are no longer a clean test. MIDAS scores reported before this section were on all 1,380 images; from here `midas_test` means the 30%.
- DDI stays whole and untouched. It is now the only test set from a source no model has trained on or been chosen on.
- `train`, `val`, `test` and the other sets do not change. The rebuild is checked against the previous manifest.
- A caution on reading results: a model trained on `midas_train` has seen that hospital's photos, so a gain on `midas_val` or `midas_test` shows it learned that source. Only a gain on DDI shows it travels to a source it has never seen.

**Idea J** (its config was removed with it; idea O's config holds the same recipe): checkpoint 1's recipe with `midas_train` added to the training images. In the balanced drawing MIDAS counts as a third dataset. Three seeds, then test-time flips and the seed average, the same treatment as checkpoint 2.

**How it is judged,** each time by paired comparison with checkpoint 2:

1. On `midas_val`. It passes if the gain in AUC is supported (the interval is above zero) and the within-source AUC on our own `val` is not lower by a supported amount.
2. If it passes, it is scored once on `test`, `midas_test`, DDI and DermaCon-IN.
3. It becomes checkpoint 3 if the gain is supported on `midas_test` or DDI and nothing is lower by a supported amount on `test` or DDI. Where the gain is will be stated with it.

If it fails step 1, it is discarded and nothing new is scored on the test sets.

**Outcome of idea J (13 Nov 2026): failed step 1, not scored on any test set.** Paired comparison with checkpoint 2, both after test-time flips and the seed average:

| Set | Score | Checkpoint 2 | Idea J | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` | AUC | 0.689 | 0.782 | +0.093 [+0.062, +0.121] | yes |
| `val` | AUC | 0.901 | 0.893 | −0.008 [−0.014, −0.001] | yes |
| `val` | Within-source AUC | 0.888 | 0.876 | −0.012 [−0.024, −0.001] | yes |

The gain on `midas_val` is large and supported, but the within-source AUC on our own `val` is lower by a supported amount, which the rule does not allow. The rule is applied as written.

What the training logs show (validation only): seeds 0 and 1 ran all 60 epochs and reached 0.867 and 0.866 within-source AUC on `val`, against 0.869, 0.872 and 0.865 for checkpoint 1's seeds. Seed 2 was stopped by the patience rule at epoch 40 with its best at epoch 25 (0.846), while the learning rate was still high; no other seed in the project has stopped before epoch 46. Most of the loss at home is that one run.

## 12. Idea O: idea J with early stopping off (13 Nov 2026)

Written before idea O was trained.

**Why.** Idea J did what it was meant to do on the outside validation photos, and lost at home mainly because one seed was cut short. Early stopping saves time; it is not part of what makes the model good, and the best weights are kept whichever epoch they come from. Turning it off removes that source of chance.

**Idea O** (`configs/skinnet-c3.yaml`): idea J with `patience: 0`, so every seed trains for all 60 epochs. Seeds 0 and 1 of idea J already ran all 60 epochs, so with early stopping off they would be the same runs; they are reused, and only seed 2 is trained again. Then test-time flips and the seed average, as before.

**How it is judged:** exactly as idea J (section 11), by paired comparison with checkpoint 2. Step 1 on `midas_val` and `val`; only if it passes, scored once on `test`, `midas_test`, DDI and DermaCon-IN; checkpoint 3 if the gain is supported on `midas_test` or DDI and nothing is lower by a supported amount on `test` or DDI.

**Cautions recorded now.**

- This idea was thought of after seeing idea J's validation result. That is what validation sets are for, and no test set has been looked at for J or O. But `val` and `midas_val` have now been used twice for this choice, so step 3 on the test sets is the real check.
- If idea O also fails step 1, the trade is real and not chance: MIDAS training photos help on MIDAS and cost a little at home. Whether that trade is worth taking is then a decision for the team, not something to settle by trying more variants.
- As for J: O has seen MIDAS's hospital. Only a gain on DDI shows it travels.

**Outcome of idea O (13 Nov 2026): the gain conditions are met, the no-loss condition is not; checkpoint 3 is not declared and the decision is with the team.** Seed 2 with early stopping off reached 0.875 within-source AUC on `val` (best at epoch 47). Paired comparison with checkpoint 2:

| Set | Score | Checkpoint 2 | Idea O | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.689 | 0.793 | +0.104 [+0.071, +0.134] | yes |
| `val` | AUC | 0.901 | 0.898 | −0.003 [−0.008, +0.003] | no |
| `val` | Within-source AUC | 0.888 | 0.882 | −0.006 [−0.014, +0.002] | no |
| `midas_test` | AUC | 0.645 | 0.698 | +0.053 [+0.020, +0.089] | yes |
| DDI | AUC | 0.629 | 0.657 | +0.028 [+0.007, +0.048] | yes |
| DermaCon-IN tumours | AUC | 0.826 | 0.841 | +0.015 [−0.058, +0.090] | no |
| `test` | AUC | 0.904 | 0.898 | −0.007 [−0.012, −0.001] | yes |
| `test` | Within-source AUC | 0.883 | 0.876 | −0.007 [−0.020, +0.006] | no |

- Step 1 passed: the gain on `midas_val` is supported and nothing on `val` is lower by a supported amount. Idea O was then scored once on the test sets.
- Step 3, gain: supported on `midas_test` and on DDI. DDI is a source no model here has trained on or been chosen on, so this is the first gain in the project shown to travel.
- Step 3, no loss: AUC on our own `test` is lower by 0.007, and the interval just excludes zero. The rule says nothing may be lower by a supported amount on `test`, so by the rule as written idea O is not a checkpoint.
- The rule is not changed here. Keeping idea O would be a change of rule after the result, as with checkpoint 2, and is for the team to decide. Until then checkpoint 2 stays the current model and idea O's runs are kept.

**Decided 13 Nov 2026: idea O is checkpoint 3.** The team accepted a loss of 0.007 AUC on our own test split for supported gains on DDI (+0.028) and the MIDAS test part (+0.053). This is a decision taken after seeing the result, not idea O passing the rule as written. Idea J's runs and config are deleted. New ideas are compared with checkpoint 3, train for all 60 epochs (`patience: 0`), and are chosen on `midas_val` together with our own `val`.

## 13. Idea I: SCIN photos in training (13 Nov 2026)

Written before the photos were downloaded and before idea I was trained.

**Why.** Checkpoint 3 showed that photos from a new kind of source are what moves the outside scores. Its remaining weakness outside is false alarms: at the fixed threshold it calls 67% of DDI's non-cancers and 28% of DermaCon-IN's everyday conditions cancer. SCIN is about 5,000 cases of everyday skin conditions photographed by the public on their phones, the kind of photo a real user would send.

**What is used.**

- Only cases the dermatologists could grade, and only as not cancer. SCIN has no biopsies, so its few suspected cancers are not trusted as cancers.
- A case is left out if a cancer, a pre-cancer or an uncertain growth appears anywhere in its list of possible diagnoses, however low its weight.
- One photo per case (the first), so SCIN does not swamp the other sources. About 3,000 photos.
- Training only. SCIN is kept outside the manifest like Derm7pt was, so every validation and test image stays exactly the same. Photos that near-duplicate a validation or test image are dropped.

**Idea I** (`configs/skinnet_scin.yaml`): checkpoint 3's recipe with these photos added. Three seeds, 60 epochs, test-time flips, seed average.

**A risk named now.** SCIN adds phone photos that are all not cancer. The first baseline learned "phone photo means cancer" from PAD-UFES-20; this could teach the opposite shortcut. It would show as more missed cancers and a lower within-source AUC on PAD-UFES-20.

**How it is judged,** by paired comparison with checkpoint 3:

1. On validation (`val` and `midas_val`). SCIN's benefit is on everyday photos, which no validation set outside our own has, so this step only checks for harm: it passes if neither the within-source AUC on `val` nor the AUC on `midas_val` is lower by a supported amount.
2. If it passes, it is scored once on `test`, `midas_test`, DDI and DermaCon-IN.
3. It becomes checkpoint 4 if at least one of these holds: a supported gain in AUC on DDI; a supported gain in AUC on `midas_test`; or the share of DermaCon-IN everyday conditions called cancer falls by at least 5 points with its interval clear of checkpoint 3's. And none of the AUCs on `test`, `midas_test` or DDI is lower by a supported amount.

If it fails step 1 it is discarded and not scored on the test sets. If it meets the gain condition but narrowly misses the no-loss one, the result is reported and the team decides, as with checkpoint 3.

**Outcome of idea I (13 Nov 2026): failed step 1, discarded, not scored on any test set.** 2,907 SCIN photos were used (2 dropped as near-duplicates of evaluation images); the training set was 16,021 images. Paired comparison with checkpoint 3, both after test-time flips and the seed average:

| Set | Score | Checkpoint 3 | Idea I | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.793 | 0.729 | −0.064 [−0.096, −0.032] | yes |
| `val` | AUC | 0.898 | 0.913 | +0.015 [+0.008, +0.023] | yes |
| `val` | Within-source AUC | 0.882 | 0.902 | +0.020 [+0.006, +0.037] | yes |

The AUC on `midas_val` is lower by a supported amount, which the rule does not allow. The gain on our own `val` is real but is the kind of gain the project has learned not to trust on its own: `val` is full of everyday conditions like SCIN's, and MIDAS is all tumours. Its config and runs are deleted. The SCIN photos stay on disk and the code that lists them stays in `data/extra.py`.

## 14. Ideas M and N: a larger network, a longer schedule (13 Nov 2026)

Written before either was trained.

**Why.** Both ask whether checkpoint 3 is held back by the model and its training and not only by the data. Neither needs new data, and both can now be chosen on the MIDAS validation part.

- **Idea M** (`configs/skinnet-idea-m.yaml`): the same network at twice the width (64 channels in the first stage, not 32), which is about four times the parameters (about 11 million against 2.8 million). Everything else as checkpoint 3.
- **Idea N** (`configs/skinnet-idea-n.yaml`): the same network trained for 120 epochs, not 60, with the same warm-up and the cosine schedule stretched to match. Everything else as checkpoint 3.

Each gets three seeds, test-time flips and the seed average, and is judged on its own by paired comparison with checkpoint 3:

1. On validation. It passes if the gain is supported on the AUC of `midas_val` or on the within-source AUC of `val`, and neither of those two is lower by a supported amount.
2. An idea that passes is scored once on `test`, `midas_test`, DDI and DermaCon-IN. If both pass, only the one with the higher AUC on `midas_val` is scored; the other is kept as a later idea to try on top of the new checkpoint.
3. It becomes checkpoint 4 if the gain is supported on at least one of `test`, `midas_test` or DDI and none of those three is lower by a supported amount. Where the gain is will be stated with it.

An idea that fails step 1 is discarded and not scored on the test sets. If one meets the gain condition and narrowly misses the no-loss one, it is reported and the team decides.

**Expectation recorded now.** Idea E (larger images) suggested detail is not the limit, and 13,000 images is little for an 11-million-parameter network trained from scratch, so M may overfit and do no better. N is the more likely of the two to help, since two of checkpoint 3's three seeds had their best epoch after epoch 45.

**Outcome of idea M (14 Nov 2026): failed step 1, discarded, not scored on any test set.** 11.2 million parameters; the seeds' best epochs were 41, 42 and 59 (within-source AUC on `val` 0.866, 0.866, 0.863, against checkpoint 3's 0.867, 0.866, 0.875). Paired comparison with checkpoint 3, both after test-time flips and the seed average:

| Set | Score | Checkpoint 3 | Idea M | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.793 | 0.773 | −0.020 [−0.037, −0.002] | yes |
| `val` | AUC | 0.898 | 0.897 | −0.001 [−0.007, +0.005] | no |
| `val` | Within-source AUC | 0.882 | 0.885 | +0.004 [−0.006, +0.013] | no |

No gain anywhere and a supported loss on `midas_val`, as expected: four times the parameters on the same 13,000 images learns no more that carries to other hospitals. Its config and runs are deleted. The width is not what holds the network back.

**Outcome of idea N, step 1 (14 Nov 2026): passed.** The seeds' best epochs were 80, 107 and 91 (within-source AUC on `val` 0.897, 0.898, 0.895), all after checkpoint 3's whole schedule had ended. Paired comparison with checkpoint 3, both after test-time flips and the seed average:

| Set | Score | Checkpoint 3 | Idea N | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.793 | 0.795 | +0.002 [−0.037, +0.042] | no |
| `val` | AUC | 0.898 | 0.930 | +0.032 [+0.023, +0.042] | yes |
| `val` | Within-source AUC | 0.882 | 0.921 | +0.039 [+0.024, +0.056] | yes |

The gain on `val` is supported and `midas_val` is not lower, so by rule 1 N goes on to the test sets. The gain is at home only so far: on the outside validation part it is flat. N was then scored once on the test sets.

**Outcome of idea N, step 3 (14 Nov 2026): failed, discarded.** Scored once on the test-only sets (ledger, 14 Nov):

| Set | Score | Checkpoint 3 | Idea N | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `test` | AUC | 0.898 | 0.930 | +0.032 [+0.022, +0.041] | yes |
| `test` | Within-source AUC | 0.876 | 0.924 | +0.048 [+0.030, +0.071] | yes |
| `midas_test` | AUC | 0.698 | 0.733 | +0.034 [−0.010, +0.075] | no |
| DDI | AUC | 0.657 | 0.592 | −0.064 [−0.096, −0.035] | yes |
| DermaCon-IN tumours | AUC | 0.841 | 0.814 | −0.028 [−0.175, +0.097] | no |

The gain condition is met on `test`; the no-loss condition fails on DDI by a wide margin, not narrowly, so this is not a case for the team and N is discarded. Its config and runs are deleted. The reading: the second half of the longer schedule taught the network more about our two home datasets (the largest gain at home in the project) and less that holds on a hospital it has never seen. DDI falls below checkpoint 2 (0.629) and the baseline (0.65). It also answers the question from section 16: more training on the same photos widens the gap to other hospitals, it does not close it.

**A suggestion for the team (adopted the same day, section 19).** Twice now (ideas I and N) a gain on `val` alone has come with a loss outside: `val` is two datasets we train on, and it rewards learning them better. Step 1 could require a supported gain on `midas_val` and not only on `val`, which would have stopped N before it spent a look at the test sets. Changing the rule is the team's call.

**Renamed 13 Nov 2026.** The kept models are now named after their checkpoint: `skinnet` (baseline), `skinnet-c1`, `skinnet-c2`, `skinnet-c3`. Configs, run folders, result folders, the test-set ledger and every mention in the documents, including earlier sections of this file, use the new names. Only names changed: no model was retrained or re-scored. The old names were `skinnet_src` (now `skinnet-c1`), `skinnet_src_tta_ens` (`skinnet-c2`) and `skinnet_midas_full_tta_ens` (`skinnet-c3`); git history before this date uses them.

## 15. Naming rule (13 Nov 2026)

One name per model, used for its config, its run folder and its result folder.

| What | Name | Example |
| --- | --- | --- |
| Baseline | `skinnet` | `configs/skinnet.yaml`, `runs/skinnet/`, `results/skinnet/` |
| A checkpoint | `skinnet-c<number>` | `skinnet-c3` |
| An idea being tested | `skinnet-idea-<letter>` | `skinnet-idea-n` |

- A run folder holds the seeds (`s0`, `s1`, `s2`) and, once built, their average (`avg`). There are no separate folders for steps on the way to a model: no `-train`, `-tta` or `-ens` names.
- An idea that is kept is renamed to the next checkpoint name everywhere (config, runs, results, ledger, documents). An idea that is discarded is deleted.
- Git tags stay `checkpoint-<number>`.
- **Added 20 Nov 2026 (team):** a reference model, one that only measures something and is never a candidate (idea P's ResNet-18s), is named `ref-<what it is>`: `ref-resnet18-scratch`, `ref-resnet18-imagenet`, `ref-resnet18-imagenet-lr1e4` (until 20 Nov without the `ref-`). The PanDerm track (section 24) follows the same pattern as SkinNet: `panderm` (its starting point), `panderm-c<number>` (its checkpoints), `panderm-idea-<letter>` (its ideas). Its runs are renamed to it once idea PF's first step ends.
- No model was retrained or re-scored for this change. `evaluate` and `compare` read `avg` when it exists, and the seeds otherwise. The pause on 13 Nov stopped ideas N and M before any seed finished; both start again from seed 0.
- Logs sit inside the model's run folder (`runs/<name>/logs/s0.log`, `final.log`, `evaluate.log`). `runs/` holds only model folders and the test-set ledger; `results/` holds only one folder per kept model.

## 16. Ideas P and Q, and a wider idea L (14 Nov 2026)

Written before any of them was started.

**Why.** Our supervisor's comment: after 60 epochs the outside scores are still about 0.6, which is poor. Checkpoint 3's training logs say the cause is not too little training. On our own validation split the within-source AUC was still rising at epoch 50 (about 0.85 to 0.87) and flattens only as the cosine schedule takes the learning rate to zero. The training loss ends at about 0.48, against a floor of about 0.20 with label smoothing 0.1, so the network is not memorising its training set. At home it scores about 0.90; on photos from other hospitals 0.66 (DDI) and 0.70 (MIDAS test). The gap is how far our training photos are from other hospitals' photos. Ideas N and M test the training side; P, Q and the wider L aim at the gap itself.

- **P. Pretrained weights.** Start from weights learned on ImageNet and train every layer. It is the largest known lever for performance on new hospitals. It breaks section 1 (from random weights, no pretrained weights anywhere in the classifier), which is the team's own rule, so it is listed and not run: the team and the supervisor decide whether the rule changes. If it does, P gets its own section with its rules before anything is trained.
- **Q. More biopsy-confirmed tumour photos from new sources.** Clinical photos (not dermoscopy) of cancers and of the benign growths that resemble them, from hospitals not yet in training. Idea O is the only change so far that moved a score on a source never seen (DDI), and it did so with photos of exactly this kind; idea I showed that new photos without tumours do not help. First step: find public sets of this kind, check their licences and their overlap with our evaluation images. None is downloaded yet.
- **L, widened.** Self-supervised pretraining on our own training photos (section 14's idea L) plus unlabelled photos already on disk that no evaluation set uses: SCIN and the Derm7pt clinical photos. Never on any image of `val`, `test`, `midas_val`, `midas_test`, DDI or DermaCon-IN, or on a near-duplicate of one. It keeps the from-scratch rule: no outside weights.

**Order.** N and M first (section 14), then L, then Q. P only if the rule is changed.

**Resumed 14 Nov 2026.** N and M were restarted from seed 0, N first, M after it. They run with `scripts/run_idea.sh`, new today: it trains the seeds, averages them and compares with checkpoint 3 on `val` and `midas_val` only, so no test set is scored before step 1 is passed. The existing `scripts/run.sh` scores the test-only sets and is for a model that has passed step 1.

## 17. Idea L: self-supervised pretraining on our own photos (14 Nov 2026)

Written before the pretraining was run.

**What.** SkinNet first learns from photos without their labels, then is trained on cancer / not cancer exactly as checkpoint 3 was. The only change from checkpoint 3 is where the weights start: from this pretraining instead of at random. No outside weights are used, so section 1 still holds.

- **Pretraining** (`python -m skincancer.pretrain --config configs/skinnet-idea-l.yaml`): SimCLR. Two random views of each photo (crop, flips, rotations, brightness, contrast and saturation changes, blur; no hue change, as in training) pass through SkinNet and a small projection head; the loss pulls the two views of a photo together and pushes the other photos in the batch apart. 200 epochs, 256 photos per step, AdamW at 1e-3 with a 10-epoch warm-up and cosine decay, temperature 0.2. One pretraining (seed 0), shared by the three training seeds. The weights after the last epoch are used; nothing is chosen on a validation set.
- **Photos used:** `train` and `midas_train` of the manifest (13,114) and the extra sets on disk, Derm7pt clinical (958) and SCIN (2,907): about 17,000. Their labels are not read. The extra sets already exclude near-duplicates of every validation and test image (`data/extra.py`). The code refuses any evaluation split.
- **Training:** `configs/skinnet-idea-l.yaml`, checkpoint 3's recipe with `model.init` pointing at the pretrained backbone. Same learning rate, schedule, epochs, sampling and test-time flips, so a difference can be traced to the starting weights.
- Weights and logs stay in the model's run folder: `runs/skinnet-idea-l/pretrain/` holds the pretraining, `s0`–`s2` and `avg` the trained model, as for every model.

**How it is judged:** as ideas M and N (section 14), by paired comparison with checkpoint 3. Step 1 on `midas_val` and `val`; only if it passes, scored once on `test`, `midas_test`, DDI and DermaCon-IN; checkpoint 4 (or the next free number) if the gain is supported on at least one of `test`, `midas_test` or DDI and none is lower by a supported amount. If N or M has become checkpoint 4 by then, L is still judged against checkpoint 3 as written here, and whether to combine the two is a separate idea. (Neither did. Step 1 was tightened on 14 Nov before L was trained: it now needs a supported gain on `midas_val`. See section 19.)

**Expectation recorded now.** Self-supervised pretraining usually helps most when the unlabelled set is much larger than the labelled one. Here it is only about a third larger, so a small gain is the likely outcome. Its value for the outside sets is that it learns from SCIN's phone photos and Derm7pt without their labels steering it, which idea I showed can hurt.

**Outcome of idea L, step 1 (14 Nov 2026): failed, discarded.** Pretraining ran its 200 epochs (final loss 2.34). The three trained seeds peaked at epochs 31, 39 and 45, with within-source AUC on `val` of 0.923, 0.922 and 0.929, the highest single SkinNet seeds in the project. Paired comparison with checkpoint 3, both after test-time flips and the seed average:

| Set | Score | Checkpoint 3 | Idea L | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.793 | 0.753 | −0.040 [−0.093, +0.015] | no |
| `val` | AUC | 0.898 | 0.936 | +0.038 [+0.027, +0.049] | yes |
| `val` | Within-source AUC | 0.882 | 0.939 | +0.058 [+0.040, +0.077] | yes |

- Against checkpoint 4, reported alongside: `midas_val` 0.818 → 0.753, −0.065 [−0.111, −0.018], a supported loss; within-source AUC on `val` 0.905 → 0.939, +0.034 [+0.021, +0.048], a supported gain.
- Section 19's corrected step 1 needs a rise on both validation sets. `midas_val` is 0.040 lower against checkpoint 3, not narrowly, and lower by a supported amount against the current checkpoint. So this is not a case for the team: L is discarded and was not scored on any test set.
- Its config and runs, the pretrained weights included, are deleted. The pretraining code (`src/skincancer/pretrain.py`) stays, so pretraining can be tried again without rewriting it.
- The reading: pretraining on photos that are mostly our two home datasets taught the network those datasets better. It is the largest gain at home of any SkinNet, and, after ideas I and N, the third time a large gain at home has come with a loss at an outside hospital. The ImageNet ResNet-18 reference (section 19, decision 2) shows the same pattern (+0.040 on `val`, −0.063 on `midas_val` against checkpoint 4). The expectation above, a small gain, was wrong in both directions.

## 18. Idea Q: MILK10k's biopsied clinical photos in training (14 Nov 2026)

Written after the photos were downloaded and before idea Q was trained. No model has been trained or scored with them.

**The source.** MILK10k (ISIC archive, collection 425; the MILK study team; CC BY-NC 4.0): 5,240 lesions, each with one clinical close-up and one dermoscopic photo, from five centres: Medical University of Vienna, Ankara University, the University of Queensland's Mayne Academy of General Practice, Memorial Sloan Kettering, and a practice in Skopje. None is Stanford (DDI, MIDAS) or in India (DermaCon-IN), so DDI stays a source no model has seen, with no shared patients.

**What is used** (`python -m skincancer.data.extra milk10k`, written to `data/extra/milk10k.csv`):

- The clinical close-ups only, one per lesion. The dermoscopic photos are a different kind of image and are not used.
- Only the 5,016 lesions confirmed by histopathology. The 224 that rest on one clinician's opinion are left out.
- Cancer: ISIC-DX says malignant and the most specific diagnosis maps to a cancer type under the label map (BCC, SCC including in situ and keratoacanthoma, melanoma including in situ). Not cancer: ISIC-DX says benign (nevi, seborrheic and lichen-planus-like keratoses, solar lentigines, dermatofibromas and others). Actinic keratoses (123, never trained on, as in the manifest) and indeterminate lesions are left out.
- Training only, kept outside the manifest, like Derm7pt and SCIN, so every validation and test image stays the same. Photos that near-duplicate a validation or test image are dropped.
- No patient ids are published, so photos are grouped by lesion.
- **Built 14 Nov:** 4,908 photos may be trained on, 3,620 cancers (2,520 BCC, 652 SCC, 448 melanoma) and 1,288 not cancer. 332 were left out (actinic keratoses, indeterminate lesions, not biopsied, or a diagnosis outside the label map). None near-duplicates a validation or test image.

**Why it may help where earlier data did not.** It is the hard distinction, cancers against the benign growths that resemble them, all biopsied, from five new centres and in large numbers (several times MIDAS's 552). Idea O showed photos of that kind from one new hospital moved DDI; idea I showed new photos without tumours do not.

**A risk named now.** Cancers are about 70% of its photos and half of all its photos are BCC. Balanced drawing inside each dataset (checkpoint 1) removes the reward for recognising MILK10k's kind of photo, but the network may still lean towards BCC-like features.

**Idea Q** (`configs/skinnet-idea-q.yaml`, now `configs/skinnet-c4.yaml`): checkpoint 3's recipe with MILK10k added. Three seeds, 60 epochs, test-time flips, seed average. Judged exactly as ideas M, N and L (section 14), by paired comparison with checkpoint 3: step 1 on `midas_val` and `val`; only if it passes, scored once on `test`, `midas_test`, DDI and DermaCon-IN; checkpoint 4 (or the next number) if the gain is supported on at least one of `test`, `midas_test` or DDI and none is lower by a supported amount. (Step 1 was tightened on 14 Nov, before Q's result was known: it now needs a supported gain on `midas_val`. See section 19.)

**Order changed.** Q runs before L, after M (section 16 had L first). Q takes about an hour and a half, L about five and a half, and Q is the more likely of the two to move the outside scores.

## 19. Team decisions after ideas M and N (14 Nov 2026)

Decided by the team on 14 Nov, after ideas M and N (section 14) and before idea Q's result was known.

**1. Step 1 now needs a gain outside.** From idea Q on, an idea passes step 1 only if the AUC on `midas_val` is higher by a supported amount and the within-source AUC on `val` is not lower by a supported amount. A gain on `val` alone no longer passes. Steps 2 and 3 are unchanged.

- Why: the outside scores are the model's main weakness. Twice (ideas I and N) a gain on `val` came with a loss outside, and N spent a look at DDI, the only source no model has trained or been chosen on, before failing there.
- The cost, accepted: `midas_val` has 414 images, so an outside gain smaller than about 0.04 cannot be shown as supported and will be discarded with the rest.
- It applies to ideas Q and L, and to every idea after them.

**1, corrected the same day: a gain on everything.** The team's answer had been read as above; the team corrected it after Q's validation result was known, and before any test set was scored for Q. The rule meant is a gain on every set, not a supported gain on one:

- **Step 1:** higher on both `midas_val` (AUC) and `val` (within-source AUC), and neither lower by a supported amount. A gain on one set alone does not pass, so ideas I and N would still have failed (`midas_val` −0.064 and +0.002).
- **Step 3 (checkpoint):** higher on every test set, `test`, `midas_test` and DDI, at least one of them by a supported amount, and none lower by a supported amount. DermaCon-IN tumours (65 images) is reported and must not be lower by a supported amount, but is too small to need a rise.
- This replaces the version above and section 14's steps 1 and 3 for Q, L and every idea after them.

**2. Idea P runs as a reference comparison, not as a candidate.** The from-scratch rule (section 1) stays for every checkpoint. P measures what that rule costs, so the team and the supervisor can decide with numbers in hand.

- SkinNet has no ImageNet weights, so P uses torchvision's ResNet-18 (11.2 million parameters). Two models, so a difference can be put on the weights and not on the network:
  - `ref-resnet18-imagenet` (`configs/ref-resnet18-imagenet.yaml`): starts from torchvision's ImageNet weights (IMAGENET1K_V1), every layer trained.
  - `ref-resnet18-scratch` (`configs/ref-resnet18-scratch.yaml`): the same network from random weights.
- Both use checkpoint 3's recipe unchanged (MIDAS training share, balanced drawing inside each dataset, AdamW at 1e-3 with warm-up and cosine decay, 60 epochs, test-time flips, three seeds and their average), so the only difference from checkpoint 3 is the network, and between the two only the starting weights. The learning rate is not tuned for the ImageNet start; if it is lower than expected, that is the first thing to check.
- Both are scored once on every set, test sets included, whatever their validation scores: they are measurements, not candidates. They are reported next to checkpoint 3 with the paired comparison and are never renamed to a checkpoint. They are named after their network, not `skinnet-`, because they are not SkinNet.
- The images are normalised with ImageNet's mean and spread already (`data/dataset.py`), so nothing else changes.
- They run after idea L.

**3. Idea K only if Q helps.** ISIC 2019 is downloaded and tried only if idea Q becomes a checkpoint with a supported gain on `midas_test` or DDI. A gain at home alone does not count. If Q fails, K is dropped.

**Outcome of idea Q, step 1 (14 Nov 2026): passes under the corrected rule (section 19, "a gain on everything"): higher on both validation sets. It would have failed the first reading of section 19.** 18,022 training images (4,908 from MILK10k). The seeds ended at within-source AUC on `val` 0.876, 0.894, 0.881. Paired comparison with checkpoint 3, both after test-time flips and the seed average:

| Set | Score | Checkpoint 3 | Idea Q | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.793 | 0.818 | +0.026 [−0.005, +0.059] | no |
| `val` | AUC | 0.898 | 0.900 | +0.002 [−0.007, +0.010] | no |
| `val` | Within-source AUC | 0.882 | 0.905 | +0.024 [+0.009, +0.039] | yes |

- Under the first reading of section 19 (a supported gain on `midas_val`) Q would have failed: the gain there is +0.026 and its interval reaches below zero. Under the corrected rule it passes: it is higher on both validation sets, and it is the only idea since checkpoint 3 to rise on both at once.
- Q was then scored once on the test sets.

**Outcome of idea Q, step 3 (14 Nov 2026): narrowly misses "higher on every test set" (DDI −0.007, not supported); reported, the team decides.** Scored once on the test-only sets (ledger, 14 Nov):

| Set | Score | Checkpoint 3 | Idea Q | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `test` | AUC | 0.898 | 0.903 | +0.005 [−0.003, +0.013] | no |
| `test` | Within-source AUC | 0.876 | 0.907 | +0.031 [+0.014, +0.052] | yes |
| `midas_test` | AUC | 0.698 | 0.750 | +0.052 [+0.004, +0.098] | yes |
| DDI | AUC | 0.657 | 0.650 | −0.007 [−0.034, +0.021] | no |
| DermaCon-IN tumours | AUC | 0.841 | 0.818 | −0.023 [−0.088, +0.030] | no |

- Supported gains on MIDAS test (the largest outside gain since checkpoint 3) and on our own test split; nothing lower by a supported amount anywhere.
- DDI is flat: 0.007 lower, well inside its interval. Under section 19's corrected step 3 (higher on every test set) that misses; under section 18's rule as first written (a supported gain on one, no supported loss) Q would be checkpoint 4.
- **Team decision, 14 Nov: accepted. Idea Q is checkpoint 4 (`skinnet-c4`).** The miss on DDI is 0.007, well inside its interval, and nothing is lower by a supported amount anywhere; the team took it as the narrow miss section 14 leaves to the team. Config, runs, ledger rows and results were renamed to `skinnet-c4`; git tag `checkpoint-4`. New ideas are compared with checkpoint 4. Idea L, specified against checkpoint 3 before checkpoint 4 existed, is judged against checkpoint 3 as written in section 17, and its comparison with checkpoint 4 is reported alongside. By section 19, decision 3, idea K now goes ahead.

**Outcome of idea P, the reference comparison (14 Nov 2026).** Both ResNet-18 models trained with checkpoint 3's recipe, three seeds each, test-time flips and the seed average, and were scored once on every set (ledger, 14 Nov). One ImageNet seed crashed at its second epoch with a GPU (cuDNN) error and was retrained from the start; nothing else differs.

| Set | Score | Checkpoint 3 | Checkpoint 4 | ResNet-18, random start | ResNet-18, ImageNet start | ImageNet against random start [95% interval] |
| --- | --- | --- | --- | --- | --- | --- |
| `val` | Within-source AUC | 0.882 | 0.905 | 0.875 | 0.943 | +0.067 [+0.048, +0.090], supported |
| `midas_val` (outside validation) | AUC | 0.793 | 0.818 | 0.785 | 0.756 | −0.029 [−0.094, +0.036] |
| `test` | AUC | 0.898 | 0.903 | 0.883 | 0.929 | +0.047 [+0.034, +0.061], supported |
| `test` | Within-source AUC | 0.876 | 0.907 | 0.880 | 0.937 | +0.057 [+0.036, +0.081], supported |
| `midas_test` | AUC | 0.698 | 0.750 | 0.713 | 0.723 | +0.010 [−0.051, +0.069] |
| DDI | AUC | 0.657 | 0.650 | 0.664 | 0.626 | −0.037 [−0.085, +0.008] |
| DermaCon-IN tumours | AUC | 0.841 | 0.818 | 0.836 | 0.867 | +0.031 [−0.115, +0.205] |

- **The network is not the limit.** From random weights the ResNet-18, four times SkinNet's size, is no better than SkinNet anywhere: slightly lower at home (`test` AUC −0.015 against checkpoint 3, supported), within noise outside. This agrees with idea M.
- **The ImageNet start helps at home only, at this learning rate.** Against the same network from random weights it raises every home score by 0.05 to 0.07 (supported), and no outside score: MIDAS validation −0.029, MIDAS test +0.010, DDI −0.037, none supported. Against checkpoint 4 it is lower on all three outside sets (MIDAS validation −0.063, supported).
- So with checkpoint 3's recipe the from-scratch rule costs nothing outside: checkpoint 4 is ahead of the ImageNet model on MIDAS and DDI. That is not what published work leads one to expect.
- **Follow-up, decided without the team (14 Nov) under the standing instruction to pick the recommended option and record it.** Section 19 named the untuned learning rate as the first thing to check if the ImageNet start did less than expected: 1e-3 on every layer for 60 epochs may overwrite most of what ImageNet taught and leave a model that has simply learned our home datasets well. `ref-resnet18-imagenet-lr1e4` is the same model at 1e-4 (`configs/ref-resnet18-imagenet-lr1e4.yaml`), a reference like the other two: scored once on every set, never a checkpoint. It uses the GPU lane freed by the two references, alongside idea K.

**Outcome of the follow-up (15 Nov 2026).** Scored once on every set (ledger, 15 Nov):

| Set | Score | Checkpoint 4 | ImageNet, lr 1e-3 | ImageNet, lr 1e-4 | lr 1e-4 against checkpoint 4 [95% interval] |
| --- | --- | --- | --- | --- | --- |
| `val` | Within-source AUC | 0.905 | 0.943 | 0.961 | +0.056 [+0.041, +0.073], supported |
| `midas_val` (outside validation) | AUC | 0.818 | 0.756 | 0.778 | −0.041 [−0.085, +0.005] |
| `test` | Within-source AUC | 0.907 | 0.937 | 0.964 | +0.057 [+0.040, +0.077], supported |
| `midas_test` | AUC | 0.750 | 0.723 | 0.737 | −0.013 [−0.066, +0.035] |
| DDI | AUC | 0.650 | 0.626 | 0.631 | −0.019 [−0.060, +0.021] |
| DermaCon-IN tumours | AUC | 0.818 | 0.867 | 0.804 | −0.014 [−0.187, +0.143] |

- The lower rate does what fine-tuning should: it is better than lr 1e-3 everywhere at home (within-source AUC on `test` +0.027, supported) and slightly better outside (MIDAS test +0.014, DDI +0.004, neither supported). It is the best model in the project at home by a wide margin.
- It is still below checkpoint 4 on all three outside sets. So the reading of idea P stands with the learning rate checked: with our training data, ImageNet weights buy a large gain at home and none at other hospitals, and keeping the from-scratch rule has cost nothing outside so far. The outside gap is a data problem, not a starting-weights problem.

## 20. Idea K: ISIC 2019 dermoscopy in training (14 Nov 2026)

Written before the images were downloaded and before anything was trained. K goes ahead because idea Q became checkpoint 4 with a supported gain on `midas_test` (section 19, decision 3).

**The source.** The ISIC 2019 challenge training set (ISIC archive collection 65, "Challenge 2019: Training"): 25,331 dermoscopic images from HAM10000 (Vienna, Queensland), BCN_20000 (Barcelona) and Memorial Sloan Kettering. Each image carries its own Creative Commons licence (CC-0, CC BY or CC BY-NC), recorded per image in `data/extra/isic2019.csv`. Some ISIC 2019 lesions may be the same people as MILK10k's (Vienna, Queensland and MSK are in both); both are training-only, so this cannot leak into a score. No ISIC centre is Stanford or in India.

**What is used** (`python -m skincancer.data.extra isic2019`):

- Labels as MILK10k (section 18): cancer when ISIC-DX says malignant and the most specific diagnosis maps to a cancer type (melanoma, BCC, SCC); not cancer when it says benign (nevi, benign keratoses, dermatofibromas, vascular lesions); actinic keratoses and indeterminate lesions left out. Unlike MILK10k a biopsy is not required: the challenge's benign ground truth also rests on follow-up and expert consensus, which is the standard reading of this set.
- **A subsample, not the whole set.** One image per lesion, then 2,500 cancers and 2,500 not cancer drawn at random with seed 0. The balanced drawing (checkpoint 1) keeps each dataset's share of the draws, so all of ISIC 2019 would be about 57% of every epoch, all of it dermoscopy, a different kind of image from every evaluation set. Idea I showed how a large share of one new kind of photo can crowd out the hard cases. At 5,000 images ISIC 2019 is about 22% of the draws, close to MILK10k's share, and a seed takes about 40 minutes, not 70. If K helps, a larger share is a separate idea.
- Stored resized to 512 pixels on the short side (some originals are 6,000 pixels wide); training crops to 224 either way.
- Training only, outside the manifest; images that near-duplicate a validation or test image are dropped.
- **Built 14 Nov:** 5,000 photos, 2,500 cancers (1,293 melanoma, 1,003 BCC, 204 SCC) and 2,500 not cancer (2,293 nevi and other benign tumours named in the label map, 207 other benign growths). None near-duplicates a validation or test image. Licences: 4,043 CC BY-NC, 957 CC-0.

**Idea K** (`configs/skinnet-idea-k.yaml`): checkpoint 4's recipe with ISIC 2019's subsample added. Three seeds, 60 epochs, test-time flips, seed average. Judged by section 19's rule against checkpoint 4: step 1 needs a rise on both `midas_val` and `val` (within-source) with neither lower by a supported amount; then scored once on the test sets; checkpoint 5 if higher on `test`, `midas_test` and DDI, at least one by a supported amount, and none lower by a supported amount (DermaCon-IN only must not be lower by a supported amount).

**Expectation recorded now.** Dermoscopy is a close-up through a lens with polarised light: no skin texture around the lesion, no glare, a fixed distance. It is uncertain whether what the network learns there carries to phone and clinic photos. The most likely gain is on melanoma against nevi, where ISIC 2019 adds far more examples than all other sources together; the most likely harm is that it weakens what MILK10k taught.

**Order.** K trains after the queue already running (idea L, then the two ResNet-18 references), around midnight.

**Outcome of idea K, step 1 (15 Nov 2026): failed, discarded.** 23,022 training images (5,000 from ISIC 2019). The seeds peaked at epochs 48, 53 and 59, within-source AUC on `val` 0.883, 0.894 and 0.888. Paired comparison with checkpoint 4, both after test-time flips and the seed average:

| Set | Score | Checkpoint 4 | Idea K | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.818 | 0.791 | −0.028 [−0.051, −0.005] | yes, a loss |
| `val` | AUC | 0.900 | 0.893 | −0.007 [−0.016, +0.001] | no |
| `val` | Within-source AUC | 0.905 | 0.908 | +0.002 [−0.011, +0.016] | no |

- Lower on `midas_val` by a supported amount and flat on `val`: no rise on either set, so not a case for the team. K is discarded and was not scored on any test set.
- Its config and runs are deleted. The subsample stays on disk (`data/extra/isic2019.csv` and its images, git-ignored) and the download code stays in `data/extra.py`, so a different use of ISIC 2019 needs no new download.
- The reading: the expectation's "most likely harm" happened. Dermoscopy close-ups at a fifth of the draws pulled the network away from what helps on MIDAS's clinic photos and did not add anything at home. More cancers are not enough; they have to look like the photos the model will see.

## 21. Idea R: MILK10k drawn twice as often (15 Nov 2026)

Written before the run. Every idea on the team's list has been run (L, K, M, N discarded; Q is checkpoint 4; P a reference). Chosen without the team, under the standing instruction to pick the recommended option and record it, so the GPU does not sit idle overnight. The team may stop it or discard it at any point.

**Why this one.** Across ideas I, J/O, K, L, M, N, P and Q only two changes raised an outside score: MIDAS photos (checkpoint 3) and MILK10k's biopsied clinical photos (checkpoint 4). Everything that helped the network learn our home datasets better (longer training, pretraining, ImageNet weights) lost outside, and dermoscopy (K) lost too. The cheapest next step along the one direction that works is to give MILK10k more of the draws. It needs no new data.

**What.** Checkpoint 4's recipe with one change: MILK10k photos are drawn twice as often (`train.dataset_weight: {milk10k: 2}`, `configs/skinnet-idea-r.yaml`). Inside each dataset cancer and not cancer stay balanced. MILK10k's share of the draws goes from 27% to 43%; Fitzpatrick17k's from 64% to 50%, PAD-UFES-20's from 6% to 5%, MIDAS's from 3.1% to 2.4%. MIDAS is not weighted up, because `midas_val` comes from the same hospital and a gain there would not show a gain anywhere new.

**How it is judged:** section 19's rule against checkpoint 4. Step 1: higher on both `midas_val` (AUC) and `val` (within-source AUC), neither lower by a supported amount. Then scored once on the test sets: checkpoint 5 if higher on `test`, `midas_test` and DDI, at least one by a supported amount, none lower by a supported amount (DermaCon-IN only must not be lower by a supported amount).

**Expectation recorded now.** A small gain on `midas_val` and DDI is possible if MILK10k's lesions are what helped; a loss at home is likely, since Fitzpatrick17k loses a fifth of its draws. If MILK10k's gain came from simply having photos from new centres, doubling the same photos adds little.

**Outcome of idea R, step 1 (15 Nov 2026): failed, discarded.** Paired comparison with checkpoint 4, both after test-time flips and the seed average:

| Set | Score | Checkpoint 4 | Idea R | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.818 | 0.815 | −0.004 [−0.028, +0.021] | no |
| `val` | AUC | 0.900 | 0.884 | −0.016 [−0.024, −0.010] | yes, a loss |
| `val` | Within-source AUC | 0.905 | 0.894 | −0.011 [−0.022, −0.001] | yes, a loss |

- Lower at home by a supported amount and flat outside: no rise on either set, so not a case for the team. R was not scored on any test set. Its config and runs are deleted; the sampling option (`train.dataset_weight`) stays in the code.
- The reading: the second half of the expectation held. Doubling MILK10k took draws from Fitzpatrick17k and lost at home, and the same photos seen twice as often taught nothing more for MIDAS. What MILK10k gave came from having photos from new centres at all, not from how often they are drawn. The next gain outside needs new photos, not a new mix of the ones we have.

## 22. Idea S: Asan and Hallym clinical photos in training (15 Nov 2026)

Written before the run. After idea R the team asked for a search for more biopsy-confirmed clinical photos from places not yet in training, and left the choice to me ("do what's best"). The search and its candidates are recorded in docs/DATA_PLAN.md; this one was chosen because it is the largest that is open to download.

**Source.** Han et al. (J Invest Dermatol 2018), figshare 5406136, CC BY 4.0: clinical photos from Asan Medical Center (Seoul) and Hallym University (Korea). A new country and a population mostly of Fitzpatrick types III to IV, which no training set has many of. The full-size images are only available on request to Asan's ethics committee; what is published are contact sheets, one per diagnosis and split, of 99 x 99 pixel thumbnails. `python -m skincancer.data.extra asan --download` fetches the archive (checksum checked) and cuts the sheets into 16,836 single photos.

**What is used.**

- Only photos the sheets mark as biopsied: 3,982 cancers (BCC 1,234, SCC 1,231, Bowen's disease 918, melanoma 599) and 8,194 benign growths (nevi, warts, dermatofibromas, haemangiomas, seborrhoeic keratoses, lentigines, pyogenic granulomas). Bowen's disease is SCC in situ, a cancer, as in the label map. Warts and haemangiomas are benign growths outside the label map, kept as not cancer as for MILK10k. Actinic keratoses and every photo not marked biopsied are left out.
- A fixed subsample of 2,500 cancers and 2,500 not cancer (seed 0), the same size as MILK10k and idea K, so Asan is about a fifth of the draws.
- No lesion or patient ids are published, so each photo is its own group. Several photos of one lesion sit side by side on a sheet, so the 5,000 photos are fewer distinct lesions. It is training-only data, so this cannot leak into a score.
- Near-duplicates of any validation or test image are dropped (`data/extra.py`, as for every extra set).
- **Built 15 Nov:** 4,999 photos, 2,500 cancers (SCC and Bowen's disease 1,363, BCC 773, melanoma 364) and 2,499 not cancer (1,654 in the label map's benign tumours, 845 warts and haemangiomas). One photo near-duplicated a validation or test image and was dropped.

**The risk.** The photos are 99 pixels square, less than half of what training crops to (224), so they are upscaled and blurred. The network may learn little from them, or learn that blur goes with a particular mix of lesions. Balanced drawing inside each dataset means blur cannot predict the class.

**Idea S** (`configs/skinnet-idea-s.yaml`): checkpoint 4's recipe with the Asan subsample added. Judged by section 19's rule against checkpoint 4. Step 1: higher on both `midas_val` (AUC) and `val` (within-source AUC), neither lower by a supported amount. Then scored once on the test sets: checkpoint 5 if higher on `test`, `midas_test` and DDI, at least one by a supported amount, none lower by a supported amount (DermaCon-IN only must not be lower by a supported amount).

**Expectation recorded now.** Of the ideas so far only new clinical photos from new places helped outside, so this is the right kind of data. The thumbnails' size is the reason it may not work. A gain on DDI and DermaCon-IN, whose patients have darker skin than most of our training photos, is the most hoped-for result.

**Outcome of idea S, step 1 (16 Nov 2026): failed, discarded.** 4,999 Asan photos were in training. The run was paused by the team after two seeds on 15 Nov and the third seed trained from its start on 16 Nov; nothing else differs. Paired comparison with checkpoint 4, both after test-time flips and the seed average:

| Set | Score | Checkpoint 4 | Idea S | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.818 | 0.792 | −0.026 [−0.051, −0.003] | yes, a loss |
| `val` | AUC | 0.900 | 0.911 | +0.011 [+0.003, +0.019] | yes, a gain |
| `val` | Within-source AUC | 0.905 | 0.924 | +0.019 [+0.005, +0.034] | yes, a gain |

- Lower on `midas_val` by a supported amount, so the rule fails outright; a supported loss outside is not a narrow miss, and S was not scored on any test set. Its config and runs are deleted; the Asan loader and `data/extra/asan.csv` stay.
- The reading: a clear gain at home, and a loss outside as large as idea K's. The recorded risk is the likely cause: 99-pixel thumbnails scaled up to 224 are blurred, and a fifth of the draws were spent on them. Why they help on Fitzpatrick17k and PAD-UFES-20 is not established; atlas and phone photos are themselves often small or soft, which would fit. So this run says little about Korean clinical photos as such, and the full-size images remain untested. Asking Asan's ethics committee for them is the team's choice.

## 23. A PanDerm reference (15 Nov 2026)

Written before the run. The team asked to check PanDerm (Yan et al., Nature Medicine 2025), a network pretrained on about 2.1 million skin images. Like idea P it is a reference comparison only, never a checkpoint: it measures what a skin-specific pretrained network would add, where P measured ImageNet's.

**The weights.** PanDerm_Base, a ViT-B/16 (86 million parameters), from the authors' GitHub (github.com/SiyuanYan1/PanDerm). Licence CC BY-NC-ND 4.0: non-commercial academic research, with attribution. The weights are kept in `data/weights/` (git-ignored) and never redistributed; a fine-tuned copy stays local. The checkpoint fits timm's BEiT ViT-B/16 key for key, so the model is built with timm (`models.PanDermBase`) and none of the authors' code is copied. A check before training: a logistic regression on PanDerm's features of 800 training photos reaches AUC 0.911 on 600 `val` photos, against 0.665 from the same network with random weights, so the weights load correctly.

**What PanDerm was trained on, and the risk to our scores.** Its pretraining (paper, Methods) used 11 sources: in-house total-body photography, dermoscopy and clinical photos from Australian and New Zealand clinics; two in-house collections of educational clinical photos (Edu1, Edu2, about 149,000) whose sources are not named; 352,000 ISIC 2024 tiles; and pathology slides. None of our evaluation sets (DDI, MIDAS, DermaCon-IN) is named among them. The unnamed educational collections could include atlas photos, and Fitzpatrick17k, part of our `test` split, was built from online atlases. So a gain on `test` may partly be images PanDerm has seen; a gain on MIDAS, DDI or DermaCon-IN is much less exposed. The paper does not state that its evaluation sets were excluded from pretraining.

**What is trained** (`configs/panderm-base.yaml`): checkpoint 4's data and sampling (MILK10k included, balanced drawing, test-time flips, three seeds and their average), every layer trained. Changed for a pretrained ViT: learning rate 5e-5 (AdamW, weight decay 0.05), 20 epochs with 2 of warm-up, the class token after the final norm feeding the same dropout and linear head. Scored once on every set and compared with checkpoint 4 and with the ImageNet ResNet-18 at 1e-4 (section 19).

**Outcome of the PanDerm reference (16 Nov 2026).** Three seeds, test-time flips and the seed average, scored once on every set (ledger, 16 Nov). The run was paused by the team on 15 Nov after the first seed; the other two trained from their starts on 16 Nov; nothing else differs. Against checkpoint 4:

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

Against the ImageNet ResNet-18 at learning rate 1e-4 (section 19):

| Set | Score | ImageNet ResNet-18 (lr 1e-4) | PanDerm | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `val` | Within-source AUC | 0.961 | 0.967 | +0.006 [−0.002, +0.014] | no |
| `midas_val` | AUC | 0.778 | 0.835 | +0.057 [+0.017, +0.098] | yes, a gain |
| `test` | Within-source AUC | 0.964 | 0.971 | +0.007 [+0.000, +0.015] | yes, a gain |
| `midas_test` | AUC | 0.737 | 0.811 | +0.074 [+0.033, +0.113] | yes, a gain |
| DDI | AUC | 0.631 | 0.650 | +0.019 [−0.018, +0.055] | no |
| DermaCon-IN (65 photos) | AUC | 0.804 | 0.880 | +0.076 [−0.087, +0.281] | no |

- The reading: PanDerm is the first pretrained network to be ahead of checkpoint 4 outside as well as at home: MIDAS test +0.061, supported, though the interval nearly reaches zero and MIDAS validation's gain (+0.016) is not supported. On DDI it is exactly level with checkpoint 4 (0.650): two million skin images in pretraining did nothing for the set with the darkest skin. Against the ImageNet ResNet-18 it is level at home and clearly better on MIDAS, so what skin pretraining adds over ImageNet is outside performance on clinic photos. The home gain (+0.06) is the one most exposed to PanDerm having seen atlas photos in pretraining (DECISIONS.md, section 23); MIDAS and DDI are not named among its data.
- It stays a reference. The from-scratch rule and the licence (CC BY-NC-ND, non-commercial, no redistribution) both rule it out as a checkpoint. What it measures: the from-scratch rule now has a cost outside, about 0.06 on MIDAS test, and none on DDI. Whether to change the rule is the team's decision.

## 24. Two tracks: from scratch and PanDerm (17 Nov 2026)

**Decided by the team (17 Nov), after the PanDerm reference:**

- PanDerm becomes a second track that can produce a final model, beside the from-scratch one. Nothing of the from-scratch track is removed: checkpoint 4, its configs, runs and documents stay, so that either can be picked at the end.
- The licence (CC BY-NC-ND 4.0: non-commercial, no redistribution) is accepted. The weights and every network fine-tuned from them stay local and out of git.
- A downloaded network scored as it comes is not our work. The PanDerm track must carry work of our own: how it is fine-tuned, what it is trained on, and what is measured. Its weak point, DDI (0.650, level with checkpoint 4), is the first target.
- I am to choose and run the experiments without stopping to ask, and report each result.

**How the PanDerm track is judged.** The same rule as section 19, with `panderm-base` (section 23) as the track's starting point in place of checkpoint 4. Step 1 on validation only: higher on both `midas_val` (AUC) and `val` (within-source AUC), neither lower by a supported amount. Then scored once on the test sets: a new PanDerm checkpoint if higher on `test`, `midas_test` and DDI, at least one by a supported amount, none lower by a supported amount (DermaCon-IN only must not be lower by a supported amount). Narrow misses go to the team. There is no DDI validation set, so no idea is ever chosen by its DDI score: DDI is read once per idea that passes step 1. `EXP=panderm-<letter> REF=panderm-base bash scripts/run_idea.sh <letter>` runs step 1.

**Ideas, in order** (PanDerm-track letters are separate from the from-scratch ones):

- **PA, gentler fine-tuning** (`configs/panderm-a.yaml`, running). `panderm-base` trained every layer at 5e-5 from the first step, with a new random head; the first steps can damage the pretrained features that carry over to unseen clinics (Kumar et al. 2022). PA trains the head alone for 3 epochs (1e-3), then every layer for 20 with a rate that shrinks by 0.75 per block towards the input (1e-4 at the top, 2.4e-6 at the patch embedding). New options `train.head_epochs`, `train.head_lr`, `train.layer_decay`. Expectation: level at home, a small gain on MIDAS; DDI uncertain.
- **PB, a larger input** (384 px): MIDAS and DDI photos are clinic photos where the lesion is a small part of the frame. (Corrected 17 Nov: this line first said a larger input had raised SkinNet's outside scores. It had not: ideas E and F at 320 px were discarded in the third round.) Needs the position embedding recomputed for the new grid. Built on PA's recipe if PA passes, else on `panderm-base`'s.
- **PC, PanDerm Large**: the paper's main network (ViT-L), if it fits in 8 GB at batch 16 with mixed precision.
- **PD, ideas that failed from scratch but may not on a pretrained network**: the Asan thumbnails (section 22) are not among them, their fault being blur; stronger colour augmentation aimed at skin tone is, and is the one most directed at DDI.

**Outcome of idea PA (17 Nov 2026): passed step 1; on the test sets a narrow miss, put to the team.** Three seeds, test-time flips and the seed average; scored once on every test set (ledger, 17 Nov). Against `panderm-base`:

| Set | Score | PanDerm base | Idea PA | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `val` | Within-source AUC | 0.967 | 0.975 | +0.007 [+0.000, +0.015] | yes, a gain |
| `midas_val` | AUC | 0.835 | 0.842 | +0.007 [−0.018, +0.034] | no |
| `test` | AUC | 0.968 | 0.967 | −0.001 [−0.007, +0.005] | no |
| `test` | Within-source AUC | 0.971 | 0.970 | −0.002 [−0.008, +0.004] | no |
| `midas_test` | AUC | 0.811 | 0.848 | +0.037 [+0.008, +0.067] | yes, a gain |
| DDI | AUC | 0.650 | 0.650 | +0.001 [−0.024, +0.026] | no |
| DermaCon-IN (65 photos) | AUC | 0.880 | 0.933 | +0.053 [+0.000, +0.122] | no |

- Step 1 passed: higher on both validation sets. On the test sets the rule asks for higher on `test`, `midas_test` and DDI. MIDAS test is higher by a supported amount; DDI is higher by 0.001, which is nothing; `test` is lower by 0.002, not supported. So the rule is missed by 0.002 on a set where both networks are at 0.97, and that is the team's to settle. My recommendation: adopt PA as the PanDerm track's starting point, since the only supported difference on any test set is a gain outside.
- Against checkpoint 4 (from scratch): `test` within-source AUC +0.062, MIDAS test 0.750 → 0.848 (+0.098), DermaCon-IN 0.818 → 0.933 (+0.114), all supported; DDI 0.650 → 0.650, level.
- DDI did not move. Gentler fine-tuning helped where PanDerm already helped (MIDAS), not where it did not. Three networks and five recipes now sit at 0.63 to 0.65 on DDI, which points at the training data rather than at how the network is trained.
- Until the team answers, later PanDerm ideas use PA's recipe and are compared with both `panderm-base` and `panderm-a`, so that neither answer makes a run useless.

**Order changed (17 Nov): PD runs before PB and PC.** After PA, DDI is the one set no recipe has moved, and PD is the idea aimed at it. It also costs 3.5 hours where PB costs about 10 (three times the image tokens), and a larger input did not help from scratch (ideas E and F). **PD** (`configs/panderm-d.yaml`): PA's recipe with `data.colour_aug`: in training, each photo is also made darker or lighter by a gamma curve (0.7 to 1.7, half the time), its saturation and hue varied a little, and each colour channel shifted. Until now the only colour change in training was brightness and contrast. Written before the run. Expectation: level at home and on MIDAS; a small gain on DDI if its difficulty is partly skin tone and lighting, none if it is the rare diagnoses, which is the likelier half. Compared at step 1 with `panderm-a` and `panderm-base`.

**Outcome of idea PD, step 1 (17 Nov 2026): failed, discarded.** Against `panderm-a`:

| Set | Score | Idea PA | Idea PD | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` | AUC | 0.842 | 0.835 | −0.007 [−0.032, +0.018] | no |
| `val` | AUC | 0.972 | 0.972 | +0.000 [−0.004, +0.005] | no |
| `val` | Within-source AUC | 0.975 | 0.976 | +0.002 [−0.003, +0.007] | no |

- Flat at home and lower on `midas_val`, so not higher on both: step 1 fails, and PD was not scored on any test set. Against `panderm-base` it is ahead at home by a supported amount and level on `midas_val` (+0.000), which is PA's gain and nothing of its own. Its config and runs are deleted; `data.colour_aug` stays in the code.
- **A check made after it, on validation only:** AUC on `val` by Fitzpatrick skin type (the labels come with Fitzpatrick17k and PAD-UFES-20; 133 photos have none):

| Model | Types I–III (1,801 photos, 377 cancers) | Type IV (419, 46) | Types V–VI (342, 35) |
| --- | --- | --- | --- |
| Checkpoint 4 | 0.901 | 0.888 | 0.895 |
| PanDerm base | 0.965 | 0.952 | 0.963 |
| Idea PA | 0.969 | 0.976 | 0.988 |
| Idea PD | 0.971 | 0.970 | 0.982 |

- No model here is worse on types V–VI than on I–III at home, and PD is not better than PA on V–VI (difference −0.006 [−0.018, +0.002]). With 35 cancers the V–VI figures are rough, but they do not show a skin-tone gap. So DDI's low score is not explained by dark skin alone: at home, dark-skin photos are scored as well as light ones. What differs in DDI is the rest: a hospital not in training, clinic photos, and its mix of diagnoses. That is why colour augmentation had nothing to fix, and it moves the weight of the remaining ideas towards data and away from recipes.

**PB deferred, PC started (17 Nov 2026).**

- **PB (384 px) is not run on this machine.** Measured before any training: at batch 32 it needs 8.6 GB on the 8 GB GPU and takes 3.2 seconds a step, about 30 minutes an epoch. At batch 16 it would be about 13 hours for three seeds and no longer PA's recipe. With ideas E and F (320 px, from scratch) discarded, that is too much for its chance. The code for another input size stays (`data.img_size` resamples PanDerm's position embedding; tested); the config is removed.
- **PC, PanDerm Large** (`configs/panderm-c.yaml`), written before the run. The paper's network is the ViT-L/16; `PanDerm_Base` is the smaller one released later. Its published file is the whole pretraining checkpoint; only the encoder is kept (`data/weights/panderm_large.pth`, same licence, not in git) and it fits timm's BEiT ViT-L/16 key for key. PA's recipe unchanged except the layer decay, 0.86 over 24 blocks in place of 0.75 over 12, which gives the same range of learning rates. Gradient checkpointing (`model.grad_checkpointing`) brings batch 32 from 11.8 GB to 6.3 GB at the same numbers; about 10 minutes an epoch, so about 11 hours for three seeds. Compared at step 1 with `panderm-a` and `panderm-base`. Expectation: a small gain at home and on MIDAS; for DDI the check by skin type above gives no reason to expect a change.

**PC sped up after its first seed (17 Nov 2026, at the team's request).** Every PanDerm seed so far (PA's three, PC's first) picked its best epoch at epoch 6, and no epoch after 10 came within 0.006 of it. So seeds 1 and 2 of PC stop 4 epochs after their best (`train.patience: 4`): 11 epochs in place of 23, about 1.9 hours a seed in place of 3.8. The learning-rate schedule is unchanged (still planned over 23 epochs), so the network a seed keeps is the same as without stopping unless an epoch after the stop would have beaten it, which none of the four seeds shows. Seed 0 ran all 23 epochs; with patience 4 it would have kept the same network.

**For later PanDerm ideas, not this one:** during the head-only epochs the whole backbone is still run backwards, only to be updated at rate zero. Freezing it then would make those epochs about three times faster, but it also changes the optimiser's state at the start of full training, so it is a change of recipe and is kept out of PC, whose seeds must match.

**Outcome of idea PC (17 Nov 2026): passed; PanDerm checkpoint 1.** Three seeds (seed 0 all 23 epochs, seeds 1 and 2 stopped at epochs 9 and 8 by the patience above), test-time flips and the seed average; scored once on every test set (ledger, 17 Nov):

| Set | Score | PanDerm base | Idea PA | PanDerm Large (PC) | PC − base [95% interval] | PC − PA [95% interval] |
| --- | --- | --- | --- | --- | --- | --- |
| `val` | Within-source AUC | 0.967 | 0.975 | **0.983** | +0.016 [+0.008, +0.025] | +0.009 [+0.003, +0.015] |
| `midas_val` | AUC | 0.835 | 0.842 | 0.856 | +0.022 [−0.007, +0.052] | +0.014 [−0.014, +0.045] |
| `test` | AUC | 0.968 | 0.967 | **0.977** | +0.010 [+0.003, +0.016] | +0.011 [+0.003, +0.018] |
| `test` | Within-source AUC | 0.971 | 0.970 | **0.982** | +0.010 [+0.004, +0.018] | +0.012 [+0.006, +0.020] |
| `midas_test` | AUC | 0.811 | 0.848 | **0.874** | +0.063 [+0.032, +0.097] | +0.026 [−0.003, +0.057] |
| DDI | AUC | 0.650 | 0.650 | **0.686** | +0.036 [+0.007, +0.064] | +0.035 [+0.011, +0.059] |
| DermaCon-IN (65 photos) | AUC | 0.880 | 0.933 | 0.850 | −0.030 [−0.164, +0.062] | −0.082 [−0.220, +0.012] |

- Against `panderm-base`, the track's reference: higher on `test`, `midas_test` and DDI, all three by a supported amount; DermaCon-IN lower by 0.030, not supported. The rule passes outright. It passes against PA as well (`test` and DDI supported gains, `midas_test` higher but not supported, DermaCon-IN not lower by a supported amount), so the open question on PA no longer matters: PC is built on PA's recipe and beats both.
- **DDI moved for the first time.** 0.650 → 0.686, a supported gain over both earlier PanDerm runs. Every network and recipe before it sat at 0.63 to 0.65. The larger network learned more from the same pretraining, and that reached the hospital none of our models has seen. Against checkpoint 4 the DDI gain (+0.036) is not supported, as the from-scratch model's DDI score is noisier against this one.
- **DermaCon-IN is the one to watch.** 0.933 with PA, 0.850 with PC: −0.082, not supported (65 photos, 19 cancers, interval −0.220 to +0.012). The rule lets it through; it is reported, not hidden.
- **Against checkpoint 4**, the from-scratch track's best:

| Set | Score | Checkpoint 4 (from scratch) | PanDerm checkpoint 1 | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `test` | Within-source AUC | 0.907 | 0.982 | +0.074 [+0.056, +0.096] | yes |
| `midas_test` | AUC | 0.750 | 0.874 | +0.124 [+0.074, +0.174] | yes |
| DDI | AUC | 0.650 | 0.686 | +0.036 [−0.010, +0.081] | no |
| DermaCon-IN (65 photos) | AUC | 0.818 | 0.850 | +0.032 [−0.123, +0.181] | no |

- `panderm-c` is PanDerm checkpoint 1. The run keeps its name (the ledger records it under `panderm-c`); its weights are copied to `checkpoints/panderm-1/` (local, git-ignored; 3.4 GB; licence as section 23) and the code is tagged `panderm-checkpoint-1`. Later PanDerm ideas are judged against it. The from-scratch track stays at checkpoint 4.

**Idea PE: stochastic depth (17 Nov 2026).** Written before the run. Every PanDerm seed peaks at epoch 6, three epochs into full training, and gets worse after: the network fits the training images quickly. PanDerm's own fine-tuning script (its repository, `classification/script/finetune_train.sh`) regularises with drop path 0.2, mixup and cutmix; ours uses none of them. PE adds only drop path 0.2 (`model.drop_path`; each block skipped at random in training, at a rate rising linearly to 0.2 at the last block), to PanDerm checkpoint 1's recipe, everything else the same (`configs/panderm-e.yaml`). Mixup and cutmix are left for later: they change the loss, and one change at a time. Judged by section 24's rule against `panderm-c`. Expectation: a later, higher peak at home; outside, a small gain if the early peak was overfitting to our training sources.

**Training can now be paused without losing a seed (18 Nov 2026).** The team paused PE twice, and each time the seed in progress started again from its first epoch. Training now writes its whole state after every epoch (`runs/<name>/s<seed>/last.pt`: network, optimiser, schedule, early-stopping count and random-number states) and a restarted seed continues after the last finished epoch; the file is deleted when the seed ends. For PanDerm Large it is about 3.6 GB, rewritten every 10 minutes. Found while testing it: training is not repeatable bit for bit even without a pause, because the augmentation library (albumentations 2) draws from its own unseeded generator, so two runs with one seed differ from the first epoch. That was already true of every run here; the seed average and the paired comparison are what make results comparable, not identical seeds.

**Outcome of idea PE, step 1 (18 Nov 2026): failed, discarded.** Three seeds; the seeds peaked at epochs 4, 4 and 3 (PanDerm checkpoint 1's at 6, 5 and 4), so stochastic depth made the peak earlier, not later as expected. Against `panderm-c`, after test-time flips and the seed average:

| Set | Score | PanDerm checkpoint 1 | Idea PE | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` | AUC | 0.856 | 0.869 | +0.013 [−0.007, +0.032] | no |
| `val` | AUC | 0.979 | 0.976 | −0.003 [−0.007, +0.002] | no |
| `val` | Within-source AUC | 0.983 | 0.980 | −0.003 [−0.006, +0.001] | no |

- Lower at home, so not higher on both validation sets: step 1 fails, and PE was not scored on any test set. As with PD, a step 1 failure is discarded, not a narrow miss. Against `panderm-a` it is higher on `midas_val` by a supported amount (+0.027), which is mostly PC's gain. Its config and runs are deleted; `model.drop_path` stays in the code.
- The same trade as ideas I and O on the from-scratch track: a little lost at home for a little gained on MIDAS, neither supported.

**Idea PF: an average of the weights (18 Nov 2026).** Written before the run. Every PanDerm seed peaks one to three epochs into full training and declines after, and PE shows that regularising each step does not move that peak. PF keeps PanDerm checkpoint 1's recipe and changes only which weights are validated, selected and kept: a running average of the weights along the way (`train.ema_decay` 0.999 per step, so about the last 1,000 steps, two epochs, carry most of the weight), not the weights at the end of one epoch. Averaging weights along a training run is a standard way to land in a flatter region that holds up better on unseen hospitals (SWAD, Cha et al., NeurIPS 2021, built for exactly that; an exponential average is its simplest form). The average is kept in CPU memory and updated every 4 steps (`train.ema_every`; measured: 0.17 s an update, about 4% more time): a second copy of PanDerm Large on the GPU needs 8.2 GB of the 8 GB. Early stopping follows the average's validation score; everything else is unchanged (`configs/panderm-f.yaml`). Judged by section 24's rule against `panderm-c`. Expectation: a later, slightly higher peak at home; on DDI and MIDAS, if anything, a small gain.
