# Both tracks compared

Results that set SkinNet and PanDerm side by side: DDI in training, the outside test sets never trained on, leave-one-dataset-out and the confident mistakes. The rules are in [DECISIONS.md](../DECISIONS.md). Each track's own history: [SkinNet results](skinnet/RESULTS.md), [PanDerm results](panderm/RESULTS.md).

## Outside test sets never trained on (22 Nov 2026)

Five sets no model trains on or is chosen by (DECISIONS.md, section 27), scored once per model. AUC, with the rates at the 0.5 cut-off for the reported versions T and PG.

| Split | Photos (cancers) | T AUC | T missed / false alarms | PG AUC | PG missed / false alarms | PG − T (paired) |
| --- | --- | --- | --- | --- | --- | --- |
| `ext_hiba` | 279 (227) | 0.866 | 36% / 12% | 0.884 | 19% / 19% | +0.018 [−0.034, +0.072] |
| `ext_mednode` | 169 (70) | 0.786 | 51% / 17% | 0.917 | 10% / 26% | +0.131 [+0.055, +0.203] |
| `ext_asan` | 12,144 (3,982) | 0.786 | 34% / 23% | 0.919 | 17% / 14% | +0.133 [+0.126, +0.140] |
| `ext_isic2019` | 4,995 (2,498) | 0.733 | 44% / 22% | 0.831 | 27% / 21% | +0.097 [+0.084, +0.111] |
| `ext_derm7pt` | 958 (294) | 0.714 | 20% / 49% | 0.814 | 14% / 44% | +0.100 [+0.064, +0.137] |
| `ddi_hold` | 328 | 0.684 | 31% / 43% | 0.785 | 30% / 27% | +0.101 [+0.045, +0.158] |
| `ext_mcrsl` (23 Nov) | 663 (80) | 0.791 | | 0.849 | | +0.057 [−0.050, +0.150] |

- **Both tracks lose ground away from home** (`test`: T 0.902, PanDerm about 0.977). SkinNet loses more: 0.03–0.19 against PanDerm's 0.06–0.19, from a lower start.
- **PanDerm is ahead outside by about 0.10**, supported on every set but HIBA. Part of this may be pretraining that saw some of these photos.
- **DDI in training did not hurt elsewhere** for PanDerm, and on SkinNet cost MED-NODE (small) and, for T, ISIC 2019 by 0.015. All models and paired tables: DECISIONS.md, section 27, and `results/compare_ext_22nov.txt`.

## Leave-one-dataset-out, PanDerm (23 Nov 2026)

PG's recipe retrained with one source removed from training and validation, 1 seed, compared (paired) with PG on that source's own test images.

| Left out | PG | Without it | Difference | SkinNet's difference |
| --- | --- | --- | --- | --- |
| PAD-UFES-20 (236) | 0.985 | 0.967 | −0.018 [−0.047, +0.003] | −0.095 |
| MIDAS (`midas_test`, 414) | 0.865 | 0.776 | −0.090 [−0.133, −0.045] | −0.028 |
| Fitzpatrick17k (2,459) | 0.971 | 0.642 | −0.329 [−0.364, −0.295] | −0.287 |
| Fitzpatrick17k, tumours only (400) | 0.892 | 0.804 | about −0.09 | about −0.09 |

- PanDerm keeps its score at an unseen phone-photo source; SkinNet does not.
- MIDAS is hard for both without training on it, PanDerm more so relative to its trained score.
- Neither track knows a rash from a cancer without Fitzpatrick17k (PanDerm: median score for rashes 0.54, against 0.08).

## Leave-one-dataset-out, SkinNet (22 Nov 2026)

T's recipe retrained with one source removed from training and validation, 3 seeds, compared (paired) with T on that source's own test images (DECISIONS.md, section 27).

| Left out | T | Without it | Difference |
| --- | --- | --- | --- |
| PAD-UFES-20 (`test`, 236) | 0.935 | 0.841 | −0.095 [−0.142, −0.054] |
| Fitzpatrick17k (`test`, 2,459) | 0.879 | 0.592 | −0.287 [−0.325, −0.250] |
| Fitzpatrick17k, tumours only (400) | 0.768 | 0.681 | about −0.09 |
| MIDAS (`midas_test`, 414) | 0.736 | 0.708 | −0.028 [−0.050, −0.005] |

- About 0.09 is what an unseen source costs SkinNet where the comparison is clean.
- Without Fitzpatrick17k the model has never seen a rash and scores them as cancer-like (median 0.60), and it trains on 60% fewer photos.
- PAD-UFES-20's phone close-ups also help on the close-up outside sets (HIBA −0.05, Derm7pt −0.05 without them).

## Confident mistakes and low scores: T and PG (22 Nov 2026)

Checked on the saved predictions of the all-DDI versions (T, PG). No training was done.

- **Confident misses are rare.** About 1% of cancers score below 0.1 on the home sets: 5 of 478 on `test` for T and 4 for PG. On DermaCon-IN tumours, PG scores 3 of 19 cancers below 0.1; two of them are melanomas that both models miss.
- **Errors both models make confidently on `test`** (cancer below 0.2 or non-cancer above 0.8 for both): 7 cancers and 11 non-cancers, all from Fitzpatrick17k.
  - Looked at by eye (not by a dermatologist), several cancer photos show no visible tumour: a "squamous cell carcinoma" that is a scattered rash on the buttocks, a "basal cell carcinoma" that is a palm covered in pits, and another that is a whole sun-damaged back.
  - The false alarms are mostly tumour-like benign lesions, such as a large bleeding pyogenic granuloma.
  - Fitzpatrick17k labels name the condition of the page they came from, not always the lesion in the photo. Part of the confident error on `test` is label noise, which no training fixes.
- **Low scores outside are mostly an offset, and only for SkinNet.**
  - On MIDAS test, T's median score for cancers is 0.43, so most fall under the 0.5 cut-off even when ranked above benign lesions.
  - Rescaling T's scores on MIDAS validation (two numbers fitted, no network training) cuts misses on MIDAS test from 59% to 29%, with false alarms rising from 11% to 34%.
  - PG is already about calibrated there: 22% missed and 24% false alarms, versus 20% and 24% after rescaling. Its MIDAS errors are ranking errors.

## DDI in training: ideas T, U, PG and PH; checkpoint 5 and PanDerm checkpoint 2 (20–22 Nov 2026)

DDI was split by near-duplicate group, stratified by cancer and skin type, into `ddi_train` (164), `ddi_val` (164) and `ddi_hold` (328, test only) (DECISIONS.md, sections 25 and 26). The four runs differ only in how much of DDI is trained on:
- **T and PG:** each track's checkpoint with all 328 trainable DDI photos.
- **U and PH:** each track's checkpoint with 164, and `ddi_val` kept for step 1.

AUC on `ddi_hold`; each difference is against the track's checkpoint without DDI:

| Track | No DDI | Quarter + `ddi_val` (U / PH) | Half (T / PG) |
| --- | --- | --- | --- |
| From scratch | 0.641 (checkpoint 4) | 0.670, +0.029, supported | 0.684, +0.043, supported |
| PanDerm | 0.691 (PanDerm checkpoint 1) | 0.733, +0.042, supported | 0.785, +0.094, supported |

By skin type (AUC on `ddi_hold`; 25, 37 and 24 cancers in I–II, III–IV and V–VI, so each figure is about ±0.1):

| Model | I–II | III–IV | V–VI |
| --- | --- | --- | --- |
| Checkpoint 4 (no DDI) | 0.621 | 0.687 | 0.637 |
| Checkpoint 5 (U, 164 DDI photos) | 0.627 | 0.725 | 0.681 |
| T (328 DDI photos) | 0.660 | 0.710 | 0.693 |
| PanDerm checkpoint 1 (no DDI) | 0.768 | 0.833 | **0.396** |
| PanDerm checkpoint 2 (PH, 164 DDI photos) | 0.730 | 0.823 | 0.595 |
| PG (328 DDI photos) | 0.749 | 0.845 | 0.726 |

PanDerm without DDI is worse than chance on the darkest skin (0.396), and nearly all of its DDI gain is there: 0.396 → 0.595 with 164 photos and → 0.726 with 328. The from-scratch network is flatter across skin types. Computed on 22 Nov from the saved predictions.

**Why PanDerm fails on the darkest skin: mycosis fungoides.**

27 of DDI's 48 cancers on skin types V–VI are mycosis fungoides, a skin lymphoma that looks like a rash (often pale patches on dark skin). Across all of DDI, with no DDI in training:
- **PanDerm checkpoint 1 ranks mycosis fungoides below DDI's benign lesions** (AUC 0.377 against them). Checkpoint 4 is near chance (0.558).
- **With mycosis fungoides left out,** the two are level on V–VI (0.665 and 0.657).

Our training has 123 mycosis fungoides photos, all from Fitzpatrick17k and only 10 of them on V–VI, against about 8,400 non-tumour photos (mostly rashes) labelled not cancer. PanDerm learns "rash → not cancer" much better than SkinNet does: it calls 1.8% of DermaCon-IN's everyday conditions cancer, against SkinNet's 6.6%. So it confidently calls a cancer that looks like a rash not cancer.

DDI's own mycosis fungoides photos fix this quickly. On `ddi_hold` (10 such photos on V–VI), PanDerm's mycosis fungoides-vs-benign AUC goes 0.222 → 0.576 with 11 in training (PanDerm checkpoint 2) and → 0.876 with 22 (PG). Without mycosis fungoides, PanDerm's lead on lighter skin (+0.08 to +0.12) is gone on V–VI, but there are only 14 to 22 such cancers.

No run is lower than its checkpoint by a supported amount on `test`, `midas_test` or DermaCon-IN. Full tables are in DECISIONS.md, section 26, and CHECKPOINTS.md.

**Reading.**
- **DDI photos in training raise DDI on both tracks,** at no supported cost elsewhere. `ddi_hold` holds other photos from the same hospital, so this shows each model learned that hospital. It does not show the models carry over to a new one.
- **PanDerm keeps gaining from more DDI photos; the from-scratch network levels off** (+0.042 then +0.094 for PanDerm, +0.029 then +0.043 from scratch).
- **`ddi_val` agreed with `ddi_hold` on both tracks** (+0.048 and +0.086 at step 1). So a DDI validation set can stand in for the test half when ideas are judged.
- **None of the four passed step 1 as written.** `val` or `midas_val` was a little lower each time, never by a supported amount.
- **The user's decision (22 Nov):** the three-way split on both tracks. U is checkpoint 5 (`skinnet-c5`) and PH is PanDerm checkpoint 2 (`panderm-c2`). T and PG were then kept as their all-DDI versions: the same configs with `ddi_val` also in training, the figures reported for each track for now.
