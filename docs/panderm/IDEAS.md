# Ideas: PanDerm track

Every idea on the PanDerm track, kept or discarded. Rules for each are written in [DECISIONS.md](../../DECISIONS.md) before it is run. Full numbers are in [RESULTS.md](RESULTS.md); kept models in [CHECKPOINTS.md](CHECKPOINTS.md). The from-scratch track's ideas: [SkinNet ideas](../skinnet/IDEAS.md).

## PanDerm reference

- **What:** PanDerm's ViT-B/16 (Nature Medicine 2025; pretrained on about 2.1 million skin images), trained on checkpoint 4's data at learning rate 5e-5 for 20 epochs. A reference like idea P, never a checkpoint.
- **Result (16 Nov):** against checkpoint 4, within-source AUC on `test` 0.907 → 0.971 and MIDAS test 0.750 → 0.811, both supported gains; MIDAS validation 0.818 → 0.835 and DermaCon-IN 0.818 → 0.880, not supported; DDI 0.650 → 0.650, level. Against the ImageNet ResNet-18: level at home, +0.06 to +0.07 on MIDAS, supported.
- **Outcome:** a measurement, recorded in DECISIONS.md, section 23. Skin pretraining helps on MIDAS where ImageNet's did not, and does nothing for DDI.
- **Caution:** its pretraining data includes unnamed educational photo collections, so its score on our `test` split may be inflated; MIDAS, DDI and DermaCon-IN are much less exposed.
## PanDerm track (from 17 Nov)

The team made PanDerm a second track beside the from-scratch one (DECISIONS.md, section 24): both are kept, and one is picked at the end. Its ideas are judged by the same rule against `panderm`.

| Idea | What | Status |
| --- | --- | --- |
| PA | Gentler fine-tuning: head alone for 3 epochs, then a learning rate that shrinks towards the early layers | Passed step 1; on the test sets MIDAS test 0.811 → 0.848 (supported), DDI level, `test` −0.002 (not supported): a narrow miss. Its recipe is the one PC (PanDerm checkpoint 1) is built on |
| PB | 384 px input | Deferred (17 Nov): does not fit the 8 GB GPU at PA's batch size; about 13 hours at half the batch |
| PC | PanDerm Large (ViT-L/16, the paper's network) on PA's recipe | **PanDerm checkpoint 1** (17 Nov): higher on every set but DermaCon-IN; DDI 0.650 → 0.686, the first supported DDI gain |
| PH | PanDerm checkpoint 1's recipe with a quarter of DDI in training and `ddi_val` in step 1, as idea U; compared with PG on `ddi_hold` | **PanDerm checkpoint 2** (`panderm-c2`, the user's decision, 22 Nov): step 1 failed on `val` alone; `ddi_val` +0.086, `ddi_hold` 0.691 → 0.733 (both supported); below PG (0.785) by a supported amount, where section 26's rule would have kept PG; the user chose the three-way split |
| PG | PanDerm checkpoint 1's recipe with half of DDI (`ddi_train`) in training, as idea T; DDI scored on `ddi_hold` | Failed step 1 (21 Nov); retrained for a look: `ddi_hold` 0.691 → 0.785 (supported); **kept: PanDerm checkpoint 2's all-DDI version** (user, 22 Nov); new ideas are compared with PanDerm checkpoint 2 |
| PF | PanDerm checkpoint 1's recipe, validated and kept as a running average of its weights | Discarded on the test sets (20 Nov): passed step 1, level at home and on MIDAS test, DDI 0.686 → 0.668 (a supported loss) |
| PE | PanDerm checkpoint 1's recipe with stochastic depth (drop path 0.2), as in PanDerm's own fine-tuning | Discarded at step 1 (18 Nov): `val` within-source AUC 0.983 → 0.980, MIDAS validation 0.856 → 0.869, neither supported |
| PD | PA's recipe with colour augmentation aimed at skin tone and lighting | Discarded at step 1 (17 Nov): flat at home, MIDAS validation 0.842 → 0.835 |

## Keeping this file up to date

When an idea is specified, add it here with its status. When it finishes, fill in what it was compared on, the result and the outcome, and update the table at the top. New ideas take the next unused letter (the next is V; on the PanDerm track, PI).
