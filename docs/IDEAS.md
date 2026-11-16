# Ideas: the full history

Every idea tried or planned in this project, by letter, in the order it was run. An idea that works becomes a checkpoint; one that does not is discarded and its config and trained models are deleted. This file keeps the record of all of them, so the history survives the deleting.

Rules for each idea are written in [DECISIONS.md](../DECISIONS.md) before it is run. Full numbers are in [RESULTS.md](RESULTS.md). The models that were kept are described in [CHECKPOINTS.md](CHECKPOINTS.md).

## At a glance

| Letter | Idea | Date | Outcome |
| --- | --- | --- | --- |
| – | Baseline | 12 Nov 2026 | Starting point (`skinnet`) |
| A | Balanced drawing inside each dataset | 12 Nov | **Kept: checkpoint 1** (`skinnet-c1`) |
| B | Hard lesion types drawn more often | 12 Nov | Discarded |
| C | B plus image-quality augmentation | 12 Nov | Discarded |
| D | Derm7pt photos added to training | 12 Nov | Discarded |
| E | Larger images (320 px) | 12 Nov | Discarded |
| F | Derm7pt and larger images together | 12 Nov | Discarded |
| G | Average the three seeds | 12 Nov | Kept as part of G+H |
| H | Test-time flips | 12 Nov | Kept as part of G+H |
| G+H | Both together | 12 Nov | **Kept: checkpoint 2** (`skinnet-c2`) |
| J | MIDAS photos added to training | 13 Nov | Discarded; led to O |
| O | J with early stopping off | 13 Nov | **Kept: checkpoint 3** (`skinnet-c3`) |
| I | SCIN photos added as not cancer | 13 Nov | Discarded |
| N | Longer schedule (120 epochs) | 14 Nov | Discarded: big gain at home, loss on DDI |
| M | Wider network (four times the parameters) | 14 Nov | Discarded |
| Q | MILK10k's biopsied clinical photos added (five new centres) | 14 Nov | **Kept: checkpoint 4** (`skinnet-c4`) |
| L | Self-supervised pretraining on our own and other unlabelled photos | 14 Nov | Discarded: largest SkinNet gain at home, loss on MIDAS |
| P | Pretrained (ImageNet) weights | 14 Nov | Reference only: ImageNet start helps at home, not outside, at either learning rate |
| K | ISIC 2019 dermoscopy images added (5,000-image subsample) | 15 Nov | Discarded: loss on MIDAS, flat at home |
| R | MILK10k drawn twice as often | 15 Nov | Discarded: loss at home, flat on MIDAS |
| S | Asan and Hallym clinical photos added (Korea; 5,000-photo subsample of 99 px thumbnails) | 16 Nov | Discarded: gain at home, loss on MIDAS |
| – | PanDerm reference: a skin-pretrained ViT-B/16 on checkpoint 4's data | 16 Nov | Reference (never a checkpoint): ahead of checkpoint 4 at home and on MIDAS test, level on DDI |

Letters were given when an idea was first listed, not when it was run, which is why I comes after O and K is last.

## How to read the scores

- **AUC:** how well the model ranks cancers above non-cancers. 0.5 is chance, 1.0 is perfect.
- **Within-source AUC:** the AUC inside each dataset, then averaged. It cannot be raised by telling datasets apart, so it is the score used on our own validation split.
- **Supported:** the paired comparison's 95% interval for the difference excludes zero.
- **Home:** our own `val` / `test` splits (Fitzpatrick17k and PAD-UFES-20). **Outside:** MIDAS, DDI and DermaCon-IN, photos from other hospitals.

## The ideas

### Baseline (`skinnet`)

- **What:** SkinNet, our own residual CNN (2.8 million parameters), trained from random weights on 12,562 images. Every image once per epoch, class-weighted loss.
- **Result:** test AUC 0.887; MIDAS 0.60, DDI 0.65. It learned "smartphone photo means cancer": 89% false alarms on PAD-UFES-20 non-cancers.
- **Outcome:** the starting point every idea is measured from. Replaced by A.

### A. Balanced drawing inside each dataset

- **What:** during training, cancer and not cancer are drawn equally often inside each dataset, so the kind of photo no longer predicts the class.
- **Compared on:** test and outside sets, 3 seeds.
- **Result:** false alarms on PAD-UFES-20 non-cancers 89% → 63%; cancers against benign lookalikes 0.72 → 0.78. Outside sets unchanged.
- **Outcome:** kept. **Checkpoint 1** (`skinnet-c1`).

### B. Hard lesion types drawn more often

- **What:** A, plus rarer and harder lesion types drawn more often (share in proportion to the square root of their count).
- **Compared on:** validation, 1 seed.
- **Result:** within-source AUC 0.847 against A's 0.869. Fewer false alarms on lookalikes (53% against 64%) but more on everything else.
- **Outcome:** discarded. A trade, not a gain.

### C. Image-quality augmentation

- **What:** B, plus random loss of resolution and JPEG quality in training, to imitate poorer photos.
- **Compared on:** validation, 1 seed.
- **Result:** 0.853 against 0.869. Same trade-off as B.
- **Outcome:** discarded.

### D. Derm7pt photos in training

- **What:** A, plus 958 clinical photos from the Derm7pt atlas (294 cancers, 664 benign) as training-only data.
- **Compared on:** validation, 1 seed.
- **Result:** 0.857 against 0.869.
- **Outcome:** discarded. One more atlas of moles and melanomas added little the network had not seen.

### E. Larger images

- **What:** A at 320 × 320 pixels (not 224 × 224).
- **Compared on:** validation, 1 seed.
- **Result:** 0.856 against 0.869; stopped early with no gain.
- **Outcome:** discarded. Image detail is not what limits this network.

### F. Derm7pt and larger images together

- **What:** D and E combined.
- **Compared on:** validation, 1 seed.
- **Result:** 0.862 against 0.869.
- **Outcome:** discarded.

### G. Average the three seeds

- **What:** no retraining. The three seeds of checkpoint 1 each predict, and their probabilities are averaged.
- **Compared on:** validation.
- **Result:** within-source AUC 0.882 against 0.868.
- **Outcome:** kept as part of G+H.

### H. Test-time flips

- **What:** no retraining. Each model predicts on the image and its three mirror images, and the four are averaged.
- **Compared on:** validation.
- **Result:** 0.879 against 0.868.
- **Outcome:** kept as part of G+H.

### G+H. Seed average and flips together

- **Compared on:** test and outside sets, paired comparison with checkpoint 1.
- **Result:** our test split AUC 0.890 → 0.904 and false alarms 40% → 33%, a supported gain. MIDAS +0.006 and DDI +0.006, neither supported.
- **Outcome:** kept. **Checkpoint 2** (`skinnet-c2`). The rule written beforehand asked for an outside gain, and by that rule it was first discarded; the team restored it because the gain at home is real, free, and nothing is worse elsewhere.

### J. MIDAS photos in training

- **Setup first:** MIDAS was split by patient into a training part (552 images), an outside validation part (414, never trained on) and a test part (414). From here models are also chosen on the outside validation part.
- **What:** checkpoint 1's recipe with the 552 MIDAS training photos added.
- **Compared on:** validation only (ours and the MIDAS validation part), paired comparison with checkpoint 2.
- **Result:** MIDAS validation AUC 0.689 → 0.782 (supported gain), but within-source AUC on our validation split 0.888 → 0.876 (supported loss). Most of the loss was one seed stopped early at epoch 25.
- **Outcome:** discarded at the first step, not scored on test sets. Its diagnosis led to O.

### O. J with early stopping off

- **What:** J, with every seed trained for all 60 epochs.
- **Compared on:** validation, then test and outside sets, paired comparison with checkpoint 2.
- **Result:** DDI 0.629 → 0.657 and MIDAS test 0.645 → 0.698, both supported. Our test split 0.904 → 0.898, a small supported loss.
- **Outcome:** kept. **Checkpoint 3** (`skinnet-c3`). It met the gain conditions and narrowly missed the no-loss one; the team accepted the trade. The first gain shown on a source never seen (DDI).

### I. SCIN photos as not cancer

- **What:** checkpoint 3's recipe with 2,907 SCIN phone photos of everyday conditions (eczema, contact dermatitis, hives, insect bites) added as not cancer, one per case.
- **Compared on:** validation only, paired comparison with checkpoint 3.
- **Result:** our validation split within-source AUC 0.882 → 0.902 (supported gain); MIDAS validation AUC 0.793 → 0.729 (supported loss).
- **Outcome:** discarded at the first step, not scored on test sets. Easy not-cancer photos help at home and hurt on tumours. The photos stay on disk (`data/raw/scin`); a much smaller share of the draws is untested.

### N. Longer schedule

- **What:** checkpoint 3's recipe trained for 120 epochs, not 60.
- **Compared on:** validation, then test and outside sets, paired comparison with checkpoint 3.
- **Result:** validation: within-source AUC 0.882 → 0.921 (supported), MIDAS validation 0.793 → 0.795 (flat). Test sets: our test split 0.898 → 0.930 (supported gain), MIDAS test 0.698 → 0.733 (not supported), DDI 0.657 → 0.592 (supported loss).
- **Outcome:** discarded (14 Nov). The largest gain at home in the project, paid for on the hospital never seen. The seeds peaked at epochs 80 to 107, so 60 epochs did cut training short, but the extra training learns our home datasets, not skin cancer in general.

### M. Wider network

- **What:** checkpoint 3's recipe with the network at twice the width, 11.2 million parameters against 2.8 million.
- **Compared on:** validation only (ours and the MIDAS validation part), paired comparison with checkpoint 3.
- **Result:** within-source AUC on our validation split 0.882 → 0.885, not supported; MIDAS validation AUC 0.793 → 0.773, a supported loss.
- **Outcome:** discarded at the first step (14 Nov), not scored on test sets. Width is not what holds the network back: four times the parameters on the same images learns nothing more that carries to other hospitals.

### L. Self-supervised pretraining on our own and other unlabelled photos

- **What:** learn general skin features without labels, then train cancer / not cancer. Widened on 14 Nov (DECISIONS.md, section 16): besides our training photos, also the unlabelled photos on disk that no evaluation set uses (SCIN, Derm7pt clinical). Never any validation or test image. No outside weights.
- **What was trained:** DECISIONS.md, section 17: SimCLR for 200 epochs on about 17,000 photos without their labels (`src/skincancer/pretrain.py`), then checkpoint 3's recipe from those weights.
- **Compared on:** validation only (ours and the MIDAS validation part), paired comparison with checkpoint 3, and with checkpoint 4 alongside.
- **Result:** within-source AUC on our validation split 0.882 → 0.939, supported, the largest SkinNet gain at home; MIDAS validation 0.793 → 0.753, lower (−0.065 against checkpoint 4, a supported loss).
- **Outcome:** discarded at the first step (14 Nov), not scored on test sets. Pretraining on mostly our own photos teaches the network our home datasets, not other hospitals: the same pattern as ideas I and N, and as the ImageNet ResNet-18 reference. The pretraining code stays for later use.

### Q. More biopsy-confirmed tumour photos from new sources

- **What:** clinical photos (not dermoscopy) of cancers and benign lookalikes from hospitals not yet in training. Idea O is the only change that moved a score on a source never seen, and it did so with photos of this kind.
- **Source found (14 Nov):** MILK10k (ISIC archive; CC BY-NC 4.0): clinical close-ups of 5,240 lesions from Vienna, Ankara, Queensland, Memorial Sloan Kettering and Skopje, 5,016 confirmed by histopathology. None of the centres is Stanford or in India, so DDI stays unseen.
- **What was trained:** checkpoint 3's recipe with 4,908 MILK10k close-ups added (3,620 cancer, 1,288 not cancer), 18,022 training images in all (DECISIONS.md, section 18).
- **Compared on:** validation, then test and outside sets, paired comparison with checkpoint 3.
- **Result:** MIDAS validation 0.793 → 0.818 and our validation split 0.882 → 0.905 (within-source), both higher. Test sets: MIDAS test 0.698 → 0.750 and our test split 0.876 → 0.907 (within-source), both supported gains; cancers against benign lookalikes at home 0.775 → 0.831; DDI 0.657 → 0.650, flat.
- **Outcome:** kept. **Checkpoint 4** (`skinnet-c4`, 14 Nov). The rule asked for a rise on every test set and DDI was 0.007 lower, inside the noise; the team accepted it. The first new data to help on a second outside source since checkpoint 3; flat on DDI in every skin-tone group.

### P. Pretrained weights

- **What:** start from ImageNet weights and train every layer. The largest known lever for new hospitals.
- **Status:** listed on 14 Nov after the supervisor called the outside scores poor (DECISIONS.md, section 16). The team decided the same day (section 19) to run it as a reference comparison only: the from-scratch rule stays for every checkpoint, and P measures what the rule costs. Two models on checkpoint 3's recipe: torchvision's ResNet-18 from ImageNet weights (`resnet18-imagenet`) and the same network from random weights (`resnet18-scratch`), so the gain can be put on the weights and not on the network. Both are scored once on every set.
- **Result (14 Nov):**

| Set | Score | Checkpoint 3 | Checkpoint 4 | ResNet-18, random start | ResNet-18, ImageNet start | ImageNet against random start [95% interval] |
| --- | --- | --- | --- | --- | --- | --- |
| `val` | Within-source AUC | 0.882 | 0.905 | 0.875 | 0.943 | +0.067 [+0.048, +0.090], supported |
| `midas_val` (outside validation) | AUC | 0.793 | 0.818 | 0.785 | 0.756 | −0.029 [−0.094, +0.036] |
| `test` | AUC | 0.898 | 0.903 | 0.883 | 0.929 | +0.047 [+0.034, +0.061], supported |
| `test` | Within-source AUC | 0.876 | 0.907 | 0.880 | 0.937 | +0.057 [+0.036, +0.081], supported |
| `midas_test` | AUC | 0.698 | 0.750 | 0.713 | 0.723 | +0.010 [−0.051, +0.069] |
| DDI | AUC | 0.657 | 0.650 | 0.664 | 0.626 | −0.037 [−0.085, +0.008] |
| DermaCon-IN tumours | AUC | 0.841 | 0.818 | 0.836 | 0.867 | +0.031 [−0.115, +0.205] |

- **Outcome:** a measurement, not a candidate. The bigger network alone does not help. ImageNet weights raise every home score by 0.05 to 0.07 and no outside score; checkpoint 4 is ahead of the ImageNet model on MIDAS and DDI. With this recipe the from-scratch rule costs nothing outside. The learning rate (1e-3 on every layer) was not tuned for an ImageNet start, so the same model was trained at 1e-4 (`resnet18-imagenet-lr1e4`, 15 Nov): better everywhere at home (within-source AUC on our test split 0.964, the best in the project), still below checkpoint 4 on MIDAS validation (0.778), MIDAS test (0.737) and DDI (0.631). The reading stands.

### K. ISIC 2019 in training

- **What:** add about 8,500 cancers and 12,900 nevi from ISIC 2019. These are dermoscopy images, a different kind of picture from ordinary photos.
- **Status:** not started. Large download; uncertain whether it carries over to clinical photos. Team decision (14 Nov, DECISIONS.md section 19): tried only if Q becomes a checkpoint with a supported gain on MIDAS test or DDI; dropped otherwise. Q did, so K goes ahead (section 20): checkpoint 4's recipe plus a fixed subsample of 2,500 cancers and 2,500 not cancer, one image per lesion, so dermoscopy is about a fifth of the draws, not over half (`data/extra/isic2019.csv`).
- **Compared on:** validation only (ours and the MIDAS validation part), paired comparison with checkpoint 4.
- **Result:** MIDAS validation 0.818 → 0.791, a supported loss; within-source AUC on our validation split 0.905 → 0.908, flat.
- **Outcome:** discarded at the first step (15 Nov), not scored on test sets. Dermoscopy close-ups do not carry over to clinic photos and crowd out what MILK10k taught. The subsample stays on disk.
### R. MILK10k drawn twice as often

- **What:** checkpoint 4's recipe with MILK10k's photos drawn twice as often (27% of the draws → 43%). MILK10k is the only new data since checkpoint 3 that raised an outside score.
- **Chosen** without the team when the list ran out (DECISIONS.md, section 21), so the GPU was not idle overnight.
- **Compared on:** validation only, paired comparison with checkpoint 4.
- **Result:** within-source AUC on our validation split 0.905 → 0.894, a supported loss; MIDAS validation 0.818 → 0.815, flat.
- **Outcome:** discarded at the first step (15 Nov), not scored on test sets. MILK10k's gain came from having new centres' photos, not from how often they are drawn.
### S. Asan and Hallym clinical photos

- **What:** checkpoint 4's recipe with 5,000 biopsied clinical photos from two Korean hospitals added (2,500 cancers, 2,500 benign growths), about a fifth of the draws. Published only as 99 x 99 pixel thumbnails, so they are blurred when scaled up to 224.
- **Why:** chosen after a search for more biopsy-confirmed clinical photos from new places (15 Nov), the only kind of data that has raised an outside score. A new country, and mostly skin types III to IV.
- **Compared on:** validation only, paired comparison with checkpoint 4.
- **Result:** within-source AUC on our validation split 0.905 → 0.924, a supported gain; MIDAS validation 0.818 → 0.792, a supported loss.
- **Outcome:** discarded at the first step (16 Nov), not scored on test sets (DECISIONS.md, section 22). The thumbnails' blur is the likely cause; the full-size images, available only on request to Asan, are untested.

### PanDerm reference

- **What:** PanDerm's ViT-B/16 (Nature Medicine 2025; pretrained on about 2.1 million skin images), trained on checkpoint 4's data at learning rate 5e-5 for 20 epochs. A reference like idea P, never a checkpoint.
- **Result (16 Nov):** against checkpoint 4, within-source AUC on `test` 0.907 → 0.971 and MIDAS test 0.750 → 0.811, both supported gains; MIDAS validation 0.818 → 0.835 and DermaCon-IN 0.818 → 0.880, not supported; DDI 0.650 → 0.650, level. Against the ImageNet ResNet-18: level at home, +0.06 to +0.07 on MIDAS, supported.
- **Outcome:** a measurement, recorded in DECISIONS.md, section 23. Skin pretraining helps on MIDAS where ImageNet's did not, and does nothing for DDI.
- **Caution:** its pretraining data includes unnamed educational photo collections, so its score on our `test` split may be inflated; MIDAS, DDI and DermaCon-IN are much less exposed.
### PanDerm track (from 17 Nov)

The team made PanDerm a second track beside the from-scratch one (DECISIONS.md, section 24): both are kept, and one is picked at the end. Its ideas are judged by the same rule against `panderm-base`.

| Idea | What | Status |
| --- | --- | --- |
| PA | Gentler fine-tuning: head alone for 3 epochs, then a learning rate that shrinks towards the early layers | Running (17 Nov) |
| PB | 384 px input | Next |
| PC | PanDerm Large (ViT-L) | After PB, if it fits the GPU |
| PD | Colour augmentation aimed at skin tone | After PC |

## Keeping this file up to date

When an idea is specified, add it here with its status. When it finishes, fill in what it was compared on, the result and the outcome, and update the table at the top. New ideas take the next unused letter (the next is R).
