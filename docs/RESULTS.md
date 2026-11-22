# Results

Stage one: is the image a skin cancer or not? Every number is the mean of three seeds with a 95% confidence interval. The rules are in [DECISIONS.md](../DECISIONS.md); the full table is [results/skinnet/metrics.csv](../results/skinnet/metrics.csv).

## Baseline (12 Nov 2026)

SkinNet, 2.8M parameters, trained from random weights on 12,562 images (2,220 cancer) from Fitzpatrick17k and PAD-UFES-20. 60 epochs, 18 seconds per epoch on an RTX 4060. The three seeds reached validation AUC 0.883, 0.874 and 0.877.

**The model works on images like its training data and much less well on images from elsewhere.** AUC is 0.89 on our own test split, 0.65 on DDI and 0.60 on MIDAS.

| Test set | Images (cancers) | AUC | Missed cancers | False alarms |
| --- | --- | --- | --- | --- |
| Our test split | 2,695 (478) | 0.887 [0.87, 0.90] | 9% [7, 11] | 42% [40, 44] |
| DDI (Stanford, biopsy-confirmed) | 656 (171) | 0.654 [0.61, 0.70] | 14% [10, 19] | 66% [63, 70] |
| MIDAS (Stanford, biopsy-confirmed tumours) | 1,380 (802) | 0.603 [0.57, 0.64] | 32% [28, 35] | 55% [52, 59] |
| DermaCon-IN tumours (India) | 65 (19) | 0.825 [0.63, 0.96] | 16% [0, 41] | 43% [32, 56] |

Missed cancers and false alarms use one fixed threshold per seed: the one that catches 90% of the cancers in the validation split. AUC 0.5 is guessing and 1.0 is perfect.

On sets with no cancers, the share of images called cancer at that threshold:

| Set | Images | Called cancer |
| --- | --- | --- |
| DermaCon-IN everyday conditions (India) | 5,379 | 35% [34, 36] |
| Actinic keratosis (precancerous, never trained on) | 703 | 91% [90, 93] |

## What the breakdown shows

**1. The photo-type shortcut is real.** On the PAD-UFES-20 part of our test split the model calls nearly everything cancer: it misses 0.4% of cancers but raises a false alarm on 89% of the non-cancers. PAD-UFES-20 images are smartphone photos and 70% of them are cancer, so the model has learned that this kind of photo usually means cancer. This was predicted before the run (DECISIONS.md, section 4).

| Part of our test split | Images (cancers) | AUC | Missed cancers | False alarms |
| --- | --- | --- | --- | --- |
| Fitzpatrick17k | 2,459 (314) | 0.850 [0.83, 0.87] | 13% [10, 16] | 41% [39, 43] |
| PAD-UFES-20 | 236 (164) | 0.864 [0.80, 0.92] | 0.4% [0, 1] | 89% [81, 96] |

The overall 0.887 is higher than either part alone because telling the two sources apart already separates much of the cancer from the non-cancer.

**2. The model cannot yet tell a cancer from a benign tumour that resembles it.** False alarms on our test split, by kind of non-cancer:

| Kind of non-cancer | Images | False alarms |
| --- | --- | --- |
| Not a tumour (psoriasis, eczema, acne ...) | 1,810 | 38% [36, 40] |
| Other benign growths (cysts, birthmarks ...) | 249 | 51% [45, 56] |
| Benign tumours that resemble cancer (nevi, seborrheic keratosis ...) | 158 | 81% [75, 86] |

MIDAS holds only tumours, cancers against benign lookalikes, which is why it scores lowest.

**3. Common cancers are caught on familiar images, rare ones are not.** Missed cancers on our test split: BCC 4%, melanoma 5%, SCC 7%, rare cancers 39% (51 images). On MIDAS the same three common cancers are each missed about 31% of the time.

**4. DermaCon-IN tumours look better (AUC 0.825), but with 19 cancers the interval runs from 0.63 to 0.96.** It is not evidence either way.

## Reading

This is a baseline, and it behaves as expected for a small network trained from random weights on this data: a good score at home that does not travel. The two weaknesses it shows are the two that [DATA_PLAN.md](DATA_PLAN.md) targets, more hard cases (benign tumours and biopsy-confirmed cancers) and smartphone photos that are not cancer.

Training was still improving slowly when it ended (best epochs 54 to 56 of 60), so a longer schedule or a larger network may add a little. Neither would fix the shortcut.

The test-only sets have now been scored once by this model, as recorded in `runs/final_ledger.csv`.

## Second round: training the shortcut cannot exploit (12 Nov 2026)

No new data. Three variants changed only how training images are drawn and augmented; one was chosen on the validation split and only that one was scored here (DECISIONS.md, section 8).

**Choosing the variant (validation split, one seed each).** The deciding score is the AUC measured inside each dataset and averaged, which telling the datasets apart cannot raise.

| | Baseline | A: balanced inside each dataset | B: A + hard types drawn more often | C: B + quality augmentation |
| --- | --- | --- | --- | --- |
| Within-source AUC | 0.850 | **0.869** | 0.847 | 0.853 |
| AUC, cancers vs benign lookalikes | 0.713 | 0.777 | 0.790 | 0.785 |
| False alarms on benign lookalikes | 72% | 64% | 53% | 49% |
| False alarms overall | 36% | 39% | 41% | 43% |

A was chosen. B and C halve the false alarms on benign lookalikes but raise them on everything else; they were not scored on the test sets.

**Variant A against the baseline, three seeds each.**

| | Baseline | Variant A |
| --- | --- | --- |
| Our test split: AUC | 0.887 [0.87, 0.90] | 0.890 [0.87, 0.91] |
| Our test split: within-source AUC | 0.857 [0.83, 0.89] | 0.865 [0.84, 0.89] |
| Our test split: AUC, cancers vs benign lookalikes | 0.723 [0.67, 0.77] | 0.783 [0.74, 0.82] |
| False alarms, PAD-UFES-20 non-cancers | 89% [81, 96] | 63% [53, 74] |
| False alarms, benign lookalikes | 81% [75, 86] | 65% [58, 72] |
| False alarms, all non-cancers | 42% [40, 44] | 40% [38, 41] |
| Missed cancers, all | 9% [7, 11] | 8% [6, 10] |
| Missed rare cancers (51 images) | 39% [28, 50] | 22% [12, 32] |
| DDI: AUC | 0.654 [0.61, 0.70] | 0.623 [0.58, 0.66] |
| MIDAS: AUC | 0.603 [0.57, 0.64] | 0.628 [0.59, 0.66] |
| MIDAS: missed cancers / false alarms | 32% / 55% | 41% / 40% |
| DermaCon-IN tumours: AUC | 0.825 [0.63, 0.96] | 0.793 [0.62, 0.95] |
| DermaCon-IN everyday conditions called cancer | 35% | 34% |
| Actinic keratoses called cancer | 91% | 80% |

**Reading.**

- **At home it worked.** The photo-type shortcut is weaker (false alarms on PAD-UFES-20 non-cancers fell from 89% to 63%) and the model is better at the hard distinction (AUC on cancers against lookalikes up from 0.72 to 0.78), without missing more cancers.
- **Outside nothing changed.** MIDAS rose a little and DDI fell a little; both differences are inside the confidence intervals. On MIDAS the model traded false alarms for missed cancers, which is a shift of the threshold and not a better model.
- **So the shortcut was not what held the outside scores down.** The gap between 0.89 at home and 0.62 outside is mostly the difference between our training images and biopsy-confirmed hospital photos, which no reweighting of the same data can close.

Variant A's training setup is kept for the next rounds. Next: add Derm7pt to training, then larger images.

## Third round: more hard cases and larger images (12 Nov 2026)

Three variants built on checkpoint 1, compared with it on the validation split, one seed each (DECISIONS.md, section 9). None was scored on the test split or the outside sets.

| | Checkpoint 1 | D: + Derm7pt | E: 320-pixel images | F: both |
| --- | --- | --- | --- | --- |
| Within-source AUC (the deciding score) | **0.869** | 0.857 | 0.856 | 0.862 |
| AUC, cancers vs benign lookalikes | 0.777 | 0.757 | 0.745 | 0.770 |
| False alarms on benign lookalikes | 64% | 62% | 62% | 60% |
| False alarms on PAD-UFES-20 non-cancers | 58% | 51% | 56% | 53% |
| False alarms overall | 39% | 41% | 42% | 40% |
| Training images | 12,562 | 13,520 | 12,562 | 13,520 |

**No variant beat checkpoint 1,** so it stays the current model. Everything tried so far is summarised in [CHECKPOINTS.md](CHECKPOINTS.md).

**Reading.**

- Adding 958 Derm7pt photos (294 cancers, 664 benign tumours) did not help. They come from one more atlas and are almost all moles and melanomas, so they add little the network had not seen.
- Larger images did not help either. The variant at 320 pixels stopped early at epoch 56 with no gain; whatever limits the model, it is not image detail at this network size.
- The differences from checkpoint 1 (0.007 to 0.013) are about the size of the seed-to-seed spread (0.01), so "no better" is the supported conclusion, not "worse".
- Three rounds now point the same way: changes to how we use these two sources move the score on these two sources at most. The outside gap needs training images from other kinds of source, and a way to measure outside performance while choosing models. Both are in [DATA_PLAN.md](DATA_PLAN.md).

## Ideas G and H: averaging seeds and test-time flips (12 Nov 2026)

No retraining: the three models of checkpoint 1 are used differently when predicting (DECISIONS.md, section 10). G averages the three seeds into one prediction. H predicts on each image and its three mirror images and averages.

**Step 1, validation split.** Within-source AUC, with a paired 95% interval on the gain over checkpoint 1:

| | Within-source AUC | Gain over checkpoint 1 |
| --- | --- | --- |
| Checkpoint 1 | 0.868 | — |
| G: seed average | 0.882 | +0.014 [0.010, 0.018] |
| H: test-time flips | 0.879 | +0.010 [0.004, 0.016] |
| G + H | 0.888 | +0.019 [0.013, 0.026] |

All three passed the 0.005 bar. G + H was the best and was the one scored on the test split and the outside sets.

**Step 2, G + H against checkpoint 1.** The gain column is a paired comparison on the same resampled images.

| | Checkpoint 1 | G + H | Gain in AUC |
| --- | --- | --- | --- |
| Our test split: AUC | 0.890 | 0.904 | +0.015 [0.012, 0.018] |
| Our test split: within-source AUC | 0.865 | 0.883 | +0.018 [0.013, 0.025] |
| Our test split: false alarms / missed cancers | 40% / 8% | 33% / 9% | |
| False alarms on benign lookalikes | 65% | 61% | |
| MIDAS: AUC | 0.628 | 0.634 | +0.006 [−0.001, 0.014] |
| MIDAS: false alarms / missed cancers | 40% / 41% | 33% / 48% | |
| DDI: AUC | 0.623 | 0.629 | +0.006 [−0.004, 0.016] |
| DDI: false alarms / missed cancers | 64% / 16% | 59% / 21% | |
| DermaCon-IN tumours: AUC | 0.793 | 0.826 | +0.033 [−0.008, 0.084] |
| DermaCon-IN everyday conditions called cancer | 34% | 24% | |

**Outcome: checkpoint 2, after a change of rule.** The rule written beforehand required the paired interval to be above zero on MIDAS or DDI. On both it includes zero, so G + H was first discarded. The team then decided that a free, supported gain on our own test split with nothing worse anywhere counts as working, and restored it as checkpoint 2 ([CHECKPOINTS.md](CHECKPOINTS.md)). The outside numbers stand as they are: not shown to be better.

**Reading.**

- At home the gain is real: about 0.015 AUC and a sixth fewer false alarms, for no training cost.
- Outside, AUC moved by +0.006, which cannot be told from no change. The lower false-alarm rates on MIDAS and DDI come with more missed cancers, so they are a moved threshold, not a better model.
- Averaging and flipping reduce noise in what the models already know. They do not add knowledge of photos from other hospitals, which is what the outside sets need.
- It is now the standard way of predicting, and every later idea is compared after the same treatment.

## Ideas J and O: MIDAS photos in training (13 Nov 2026)

MIDAS was split by patient into three parts: 552 images that may be trained on, 414 for outside validation (never trained on, used to choose models) and 414 for testing. From here on a MIDAS test score is on those 414 images; earlier sections used all 1,380. Checkpoint 2 re-scored on the new parts: outside validation 0.689, MIDAS test 0.645.

**Idea J** is checkpoint 1's recipe with the 552 MIDAS training images added. On the outside validation part it gained +0.093 AUC, but within-source AUC on our own validation split fell by 0.012, a supported loss, so it failed the first step and was not scored on any test set. Most of that loss was one seed stopped early at epoch 25.

**Idea O** is idea J with early stopping off, so every seed trains for 60 epochs. Both compared with checkpoint 2 after test-time flips and the seed average:

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

At the threshold that catches 90% of cancers on our validation split:

| Set | Cancers missed: checkpoint 2 | Idea O | False alarms: checkpoint 2 | Idea O |
| --- | --- | --- | --- | --- |
| Our test split | 9% | 10% | 33% | 34% |
| MIDAS test | 50% | 33% | 31% | 41% |
| DDI | 21% | 16% | 59% | 67% |
| DermaCon-IN tumours | 21% | 16% | 30% | 35% |

DermaCon-IN everyday conditions called cancer: 24% → 28%.

**Reading.**

- Idea O ranks cancers above non-cancers better on both hospital test sets. The DDI gain matters most: no model has trained on DDI's source or been chosen on it.
- The MIDAS test gain is partly "has now seen this hospital", so it says less about new hospitals than the DDI gain does.
- At the fixed threshold the outside change is mostly a shift: fewer cancers missed, more false alarms. The AUC gains say the ranking improved as well, but the threshold set on our own validation split still does not carry over to other hospitals.
- At home it costs a little: test AUC −0.007, cancers against benign lookalikes 0.797 → 0.775.

**Outcome: checkpoint 3, by a decision of the team.** The rule written beforehand required a supported gain on MIDAS test or DDI, which idea O has on both, and no supported loss on our own test split, which it misses by a small margin. The team decided the trade is worth taking, since new hospitals are the open problem. See [CHECKPOINTS.md](CHECKPOINTS.md) and ([DECISIONS.md](../DECISIONS.md), section 12).

## Idea I: SCIN photos in training (13 Nov 2026)

SCIN is a public set of phone photos of everyday skin conditions (eczema, contact dermatitis, hives, insect bites). 2,907 of them, one per case, were added to checkpoint 3's training images as not cancer. Cases with a cancer or pre-cancer anywhere in the dermatologists' list were left out.

| Set | Score | Checkpoint 3 | Idea I | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.793 | 0.729 | −0.064 [−0.096, −0.032] | yes |
| `val` | AUC | 0.898 | 0.913 | +0.015 [+0.008, +0.023] | yes |
| `val` | Within-source AUC | 0.882 | 0.902 | +0.020 [+0.006, +0.037] | yes |

**Outcome: discarded at the first step.** It was not scored on any test set.

**Reading.**

- At home it is the largest gain since checkpoint 1. Our validation split holds many everyday conditions from Fitzpatrick17k, and more examples of those make them easier to tell from cancer.
- On the MIDAS validation part it gives back most of what checkpoint 3 gained (0.793 → 0.729; checkpoint 2 was 0.689). MIDAS is all tumours: the question there is cancer against a benign growth, and SCIN has no growths at all.
- The likely cause is dilution. In the balanced drawing SCIN took about 18% of every epoch, all of it easy "rash, so not cancer" examples, and the share of draws going to the hard tumour-against-tumour cases fell.
- So more photos from a new source is not enough on its own. What helped in checkpoint 3 was photos of the hard distinction from a new source.

## Ideas M and N: a wider network, a longer schedule (14 Nov 2026)

Both are checkpoint 3's recipe with one thing changed: M has the network at twice the width (11.2 million parameters against 2.8 million), N trains for 120 epochs instead of 60. Three seeds each, test-time flips and the seed average, paired comparison with checkpoint 3.

**Step 1, validation:**

| Set | Score | Checkpoint 3 | Idea M | Idea N |
| --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.793 | 0.773 (−0.020, supported loss) | 0.795 (+0.002, not supported) |
| `val` | AUC | 0.898 | 0.897 (−0.001) | 0.930 (+0.032, supported) |
| `val` | Within-source AUC | 0.882 | 0.885 (+0.004) | 0.921 (+0.039, supported) |

M failed step 1 and was not scored on any test set. N passed and was scored once on the test sets:

| Set | Score | Checkpoint 3 | Idea N | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `test` | AUC | 0.898 | 0.930 | +0.032 [+0.022, +0.041] | yes |
| `test` | Within-source AUC | 0.876 | 0.924 | +0.048 [+0.030, +0.071] | yes |
| `midas_test` | AUC | 0.698 | 0.733 | +0.034 [−0.010, +0.075] | no |
| DDI | AUC | 0.657 | 0.592 | −0.064 [−0.096, −0.035] | yes |
| DermaCon-IN tumours | AUC | 0.841 | 0.814 | −0.028 [−0.175, +0.097] | no |

**Outcome: both discarded.** M has no gain anywhere and a loss on the MIDAS validation part. N has the largest gain at home in the project and a supported loss on DDI, which the rule does not allow.

**Reading.**

- Width is not the limit. Four times the parameters on the same 13,000 images learns nothing more that carries to other hospitals.
- Length is a limit at home only. N's seeds did best at epochs 80 to 107, long after checkpoint 3's schedule had ended, and our own test split rose from 0.898 to 0.930. On DDI, a hospital no model here has seen, it fell from 0.657 to 0.592. The extra epochs taught the network our two home datasets better, and part of what it learned does not hold anywhere else.
- So the supervisor's point is right that 60 epochs was not the end of training, but more of it on the same photos widens the gap to other hospitals. The gap needs different photos or different starting weights (ideas Q, L and P), not more of the same training.

## Idea L: self-supervised pretraining (14 Nov 2026)

SkinNet first learned from about 17,000 photos without their labels (SimCLR, 200 epochs: our training photos, the MIDAS training part, Derm7pt and SCIN), then was trained with checkpoint 3's recipe from those weights. Three seeds, test-time flips and the seed average. Judged against checkpoint 3 as specified; checkpoint 4 alongside.

| Set | Score | Checkpoint 3 | Checkpoint 4 | Idea L | L against checkpoint 3 | L against checkpoint 4 |
| --- | --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.793 | 0.818 | 0.753 | −0.040 [−0.093, +0.015] | −0.065 [−0.111, −0.018], supported |
| `val` | AUC | 0.898 | 0.900 | 0.936 | +0.038 [+0.027, +0.049], supported | +0.036 [+0.024, +0.049], supported |
| `val` | Within-source AUC | 0.882 | 0.905 | 0.939 | +0.058 [+0.040, +0.077], supported | +0.034 [+0.021, +0.048], supported |

**Outcome: discarded at the first step**, not scored on any test set. The rule needs a rise on both validation sets; MIDAS validation fell.

**Reading.** The largest gain at home of any SkinNet, and a loss at the outside hospital: the same shape as ideas I and N. About three in four of the pretraining photos come from our two home datasets, so the pretraining learned what those photos look like. Starting weights that help outside would have to come from photos unlike ours, which is what the ImageNet reference (idea P) tests.

## Idea P: what ImageNet weights would add, a reference (14 Nov 2026)

Two torchvision ResNet-18 models (11.2 million parameters), checkpoint 3's recipe unchanged: one from ImageNet weights, one from random weights, so the effect of the weights can be told apart from the effect of a different, larger network. Three seeds each, test-time flips and the seed average, scored once on every set. Never a checkpoint.

| Set | Score | Checkpoint 3 | Checkpoint 4 | ResNet-18, random start | ResNet-18, ImageNet start | ImageNet against random start [95% interval] |
| --- | --- | --- | --- | --- | --- | --- |
| `val` | Within-source AUC | 0.882 | 0.905 | 0.875 | 0.943 | +0.067 [+0.048, +0.090], supported |
| `midas_val` (outside validation) | AUC | 0.793 | 0.818 | 0.785 | 0.756 | −0.029 [−0.094, +0.036] |
| `test` | AUC | 0.898 | 0.903 | 0.883 | 0.929 | +0.047 [+0.034, +0.061], supported |
| `test` | Within-source AUC | 0.876 | 0.907 | 0.880 | 0.937 | +0.057 [+0.036, +0.081], supported |
| `midas_test` | AUC | 0.698 | 0.750 | 0.713 | 0.723 | +0.010 [−0.051, +0.069] |
| DDI | AUC | 0.657 | 0.650 | 0.664 | 0.626 | −0.037 [−0.085, +0.008] |
| DermaCon-IN tumours | AUC | 0.841 | 0.818 | 0.836 | 0.867 | +0.031 [−0.115, +0.205] |

**Reading.**

- A bigger network on its own does nothing: the random-start ResNet-18 matches SkinNet within noise everywhere, as idea M found.
- ImageNet weights make the model much better on our two home datasets and no better at other hospitals. This is the same pattern as ideas I, N and L: whatever helps the network learn our home photos better does not carry outside.
- Checkpoint 4, from scratch with more hospitals' photos in training, is ahead of the ImageNet model on MIDAS (0.750 against 0.723 on its test part) and DDI (0.650 against 0.626). New photos from new places did more for the outside scores than ImageNet weights.
- The ImageNet model was trained at checkpoint 3's learning rate (1e-3 on every layer), not tuned for a pretrained start, so a third reference used ten times lower, the usual choice for fine-tuning (15 Nov):

| Set | Score | Checkpoint 4 | ImageNet, lr 1e-3 | ImageNet, lr 1e-4 | lr 1e-4 against checkpoint 4 [95% interval] |
| --- | --- | --- | --- | --- | --- |
| `val` | Within-source AUC | 0.905 | 0.943 | 0.961 | +0.056 [+0.041, +0.073], supported |
| `midas_val` (outside validation) | AUC | 0.818 | 0.756 | 0.778 | −0.041 [−0.085, +0.005] |
| `test` | Within-source AUC | 0.907 | 0.937 | 0.964 | +0.057 [+0.040, +0.077], supported |
| `midas_test` | AUC | 0.750 | 0.723 | 0.737 | −0.013 [−0.066, +0.035] |
| DDI | AUC | 0.650 | 0.626 | 0.631 | −0.019 [−0.060, +0.021] |
| DermaCon-IN tumours | AUC | 0.818 | 0.867 | 0.804 | −0.014 [−0.187, +0.143] |

- The lower rate helps at home (the best home scores in the project) and a little outside, and is still below checkpoint 4 on all three outside sets. The outside gap is about the training photos, not the starting weights.

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

## Idea S: Asan and Hallym clinical photos in training (16 Nov 2026)

Checkpoint 4's recipe with 4,999 biopsied clinical photos from two Korean hospitals added (2,500 cancers, 2,499 not cancer), about a fifth of the draws. They are published only as 99 x 99 pixel thumbnails. Three seeds, test-time flips and the seed average, paired comparison with checkpoint 4.

| Set | Score | Checkpoint 4 | Idea S | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.818 | 0.792 | −0.026 [−0.051, −0.003] | yes, a loss |
| `val` | AUC | 0.900 | 0.911 | +0.011 [+0.003, +0.019] | yes, a gain |
| `val` | Within-source AUC | 0.905 | 0.924 | +0.019 [+0.005, +0.034] | yes, a gain |

**Outcome: discarded at the first step**, not scored on any test set.

**Reading.** A clear gain at home, and a loss on MIDAS as large as idea K's. The likely cause is the size of the photos: scaled up from 99 to 224 pixels they are blurred. The full-size images, which Asan gives only on request, are untested.

## Idea R: MILK10k drawn twice as often (15 Nov 2026)

Checkpoint 4's recipe with MILK10k's share of the draws raised from 27% to 43%. Paired comparison with checkpoint 4.

| Set | Score | Checkpoint 4 | Idea R | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.818 | 0.815 | −0.004 [−0.028, +0.021] | no |
| `val` | AUC | 0.900 | 0.884 | −0.016 [−0.024, −0.010] | yes, a loss |
| `val` | Within-source AUC | 0.905 | 0.894 | −0.011 [−0.022, −0.001] | yes, a loss |

**Outcome: discarded at the first step.** Fewer Fitzpatrick17k draws cost at home; the same MILK10k photos seen more often added nothing outside.

## Idea K: ISIC 2019 dermoscopy in training (15 Nov 2026)

Checkpoint 4's recipe with 5,000 ISIC 2019 dermoscopy images added (2,500 cancers, 2,500 not cancer, one per lesion), about a fifth of the draws. Three seeds, test-time flips and the seed average, paired comparison with checkpoint 4.

| Set | Score | Checkpoint 4 | Idea K | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.818 | 0.791 | −0.028 [−0.051, −0.005] | yes, a loss |
| `val` | AUC | 0.900 | 0.893 | −0.007 [−0.016, +0.001] | no |
| `val` | Within-source AUC | 0.905 | 0.908 | +0.002 [−0.011, +0.016] | no |

**Outcome: discarded at the first step**, not scored on any test set.

**Reading.** Dermoscopy adds thousands of cancers, but they are close-ups through a lens with no skin around them. They did nothing at home and cost 0.028 on MIDAS's clinic photos. Together with idea Q, it says what kind of data helps: clinical photos of the hard cases, from new places. More cancers of a different kind of image do not.

## Idea Q: MILK10k photos in training, checkpoint 4 (14 Nov 2026)

4,908 biopsy-confirmed clinical close-ups from MILK10k (five centres: Vienna, Ankara, Queensland, Memorial Sloan Kettering, Skopje) added to checkpoint 3's training images: 3,620 cancers (2,520 BCC, 652 SCC, 448 melanoma) and 1,288 benign growths. Three seeds, test-time flips, seed average, paired comparison with checkpoint 3.

| Set | Score | Checkpoint 3 | Checkpoint 4 (idea Q) | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.793 | 0.818 | +0.026 [−0.005, +0.059] | no |
| `val` | Within-source AUC | 0.882 | 0.905 | +0.024 [+0.009, +0.039] | yes |
| `test` | AUC | 0.898 | 0.903 | +0.005 [−0.003, +0.013] | no |
| `test` | Within-source AUC | 0.876 | 0.907 | +0.031 [+0.014, +0.052] | yes |
| `midas_test` | AUC | 0.698 | 0.750 | +0.052 [+0.004, +0.098] | yes |
| DDI | AUC | 0.657 | 0.650 | −0.007 [−0.034, +0.021] | no |
| DermaCon-IN tumours | AUC | 0.841 | 0.818 | −0.023 [−0.088, +0.030] | no |

At the threshold fixed on validation: MIDAS test false alarms 41% → 29% and missed cancers 33% → 29%; at home, cancers against benign lookalikes 0.775 → 0.831.

**Outcome: checkpoint 4, by a decision of the team.** The rule (DECISIONS.md, section 19) asked for a rise on every test set; DDI was 0.007 lower, inside the noise, and nothing was lower by a supported amount.

**Reading.**

- The first idea since checkpoint 3 to rise on both validation sets, and the largest gain yet on the hard question: cancer against a benign growth that looks like one. That is what MILK10k is made of.
- It helps on MIDAS, a hospital it has not trained on more than before, so the gain is not just learning MILK10k's photos.
- DDI is flat, and skin tone does not explain it. A first guess was that MILK10k's mostly lighter-skinned centres would not help on DDI's darker skin, but by DDI's skin-tone groups (about 200 images each) checkpoint 4 is 0.676 → 0.645 on Fitzpatrick I–II, 0.679 → 0.713 on III–IV and 0.625 → 0.607 on V–VI: lower on the lightest group too, and every change within noise. Why it helps on MIDAS and not on DDI is open. One difference that may matter: MIDAS's cancers are all BCC, SCC and melanoma, the three types MILK10k has, while 40 of DDI's 171 cancers are rarer types (32 mycosis fungoides, 6 Kaposi sarcoma). In training those appear only in Fitzpatrick17k (236 photos), and adding MILK10k makes them a smaller share of what the network sees. A guess, not tested.
- It calls actinic keratoses (pre-cancers, never trained on) cancer more often (80% → 88%), in line with a training set that is half basal cell carcinoma.
