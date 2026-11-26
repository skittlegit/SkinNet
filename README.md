# SkinNet: skin cancer detection

Skin cancer detection from clinical skin photos: the model tells whether a photo shows a skin cancer. SkinNet is a small residual CNN of our own, trained from random weights on our data: no pretrained weights, no frozen layer.

The project's other model, fine-tuned from a pretrained dermatology foundation model, is in its own repository, `skittlegit/PanDerm`. Both use the same data, splits and rules; this repository's history holds the work from before they were separated (24 Nov 2026).

The model trains on 18,186 photos from five sources. It is tested on:
- held-back photos from those sources;
- MIDAS (Stanford) and DDI (Stanford, all skin types), whose other photos are partly in training;
- DermaCon-IN (India);
- six outside sets no model ever trains on: HIBA (Argentina), MED-NODE (Netherlands), MCR-SL (Norway), Derm7pt (Italy), Asan and Hallym (Korea), ISIC 2019 ([DECISIONS.md](DECISIONS.md), section 27).

This is stage one. Telling cancer types apart, and whether darker skin needs its own model, come later.

This is a research prototype, not a diagnostic tool.

**Team:** Ankita Vemavarapu, Aeleni Deepak Reddy and Ranganaini Sai Ruchir. **Supervisor:** Dr. Anjli Varghese.

**Read [DECISIONS.md](DECISIONS.md) before changing anything.** It holds the rules for labels, splits, the network, the training recipe and the metrics, and the code enforces them. It is a record in order: each section was written before the run it governs.

## The current model

Checkpoint 5, `skinnet-c5` (`configs/skinnet-c5.yaml`, 25 Nov 2026):
- **Network:** SkinNet, 2.8 million parameters, from random weights.
- **Training:** 60 epochs on 18,186 photos: Fitzpatrick17k, PAD-UFES-20, MILK10k, 552 MIDAS photos and 164 DDI photos. Cancer and not cancer are drawn equally often inside each source. Sharpness-aware minimisation (SAM) and a running average of the weights.
- **Prediction:** three seeds, averaged, each predicting on the image and its mirror images.
- **Threshold:** the one that catches 90% of cancers on `val`.

| Set | Images (cancers) | AUC [95% interval] | Missed cancers | False alarms |
| --- | --- | --- | --- | --- |
| Our test split | 2,695 (478) | 0.901 [0.883, 0.916] | 9.4% | 35.6% |
| MIDAS test | 414 (241) | 0.734 [0.669, 0.794] | 30.3% | 34.7% |
| DDI, held-out half | 328 (86) | 0.676 [0.617, 0.733] | 10.5% | 68.2% |
| DermaCon-IN tumours | 65 (19) | 0.846 [0.666, 0.972] | 15.8% | 30.4% |
| HIBA | 279 (227) | 0.849 [0.792, 0.901] | 15.9% | 32.7% |
| MED-NODE | 169 (70) | 0.823 [0.759, 0.884] | 7.1% | 59.6% |
| MCR-SL | 663 (80) | 0.805 [0.680, 0.899] | 41.2% | 13.9% |
| Derm7pt | 958 (294) | 0.712 [0.679, 0.745] | 5.4% | 76.1% |
| Asan and Hallym | 12,144 (3,982) | 0.793 [0.785, 0.802] | 10.2% | 52.5% |
| ISIC 2019 (dermoscopy) | 4,995 (2,498) | 0.729 [0.715, 0.742] | 20.5% | 48.9% |

Of DermaCon-IN's 5,379 everyday conditions, 32.9% are called cancer.

**Where it stands.** About 0.90 on photos like its training data, 0.71 to 0.85 on the seven sets from hospitals it has never trained on, and 0.68 to 0.73 on MIDAS and DDI. A threshold set on our own validation split does not carry over: outside, it misses more cancers (MCR-SL 41%) or raises more false alarms (Derm7pt 76%). Most outside gains so far came from training on photos from that hospital (MIDAS, MILK10k, DDI). Changes to training mostly helped at home only; checkpoint 5's SAM is the first that also gained on outside test sets (MED-NODE +0.052, Asan +0.009, both supported), at a cost on ISIC 2019 (−0.021). More epochs do not help outside (ideas N, AB and AC).

**Against PanDerm**, the project's other model: DermFM-Zero, fine-tuned on the same photos and splits (`skittlegit/PanDerm`). Paired on the same images, seed averages:

| Set | SkinNet c5 | PanDerm | PanDerm − SkinNet [95% interval] |
| --- | --- | --- | --- |
| Our test split | 0.901 | 0.991 | +0.090 [+0.074, +0.108] |
| MIDAS test | 0.734 | 0.863 | +0.130 [+0.072, +0.191] |
| DDI, held-out half | 0.676 | 0.787 | +0.111 [+0.047, +0.180] |
| DermaCon-IN tumours | 0.846 | 0.898 | +0.053 [−0.033, +0.141], not supported |
| HIBA | 0.849 | 0.955 | +0.105 [+0.052, +0.164] |
| MED-NODE | 0.823 | 0.951 | +0.128 [+0.069, +0.192] |
| MCR-SL | 0.805 | 0.901 | +0.096 [−0.027, +0.189], not supported |
| Derm7pt | 0.712 | 0.875 | +0.163 [+0.130, +0.197] |
| Asan and Hallym | 0.793 | 0.950 | +0.157 [+0.148, +0.164] |
| ISIC 2019 | 0.729 | 0.934 | +0.206 [+0.193, +0.218] |

- PanDerm starts from a network pretrained on about 2.1 million skin images; SkinNet has only our 18,186.
- DermFM-Zero's pretraining included Fitzpatrick17k, all of HIBA and part of ISIC 2019, so PanDerm's scores there are optimistic. On the 2,857 ISIC 2019 images it never saw, the gap is +0.173 [+0.156, +0.190].

Where to read:
- Every checkpoint, with the scores that made it one: [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md).
- Every idea by letter, kept and discarded: [docs/IDEAS.md](docs/IDEAS.md); full numbers: [docs/RESULTS.md](docs/RESULTS.md).
- Data sources, tried and planned: [docs/DATA_PLAN.md](docs/DATA_PLAN.md).

The project was restarted on 12 Nov 2026. The earlier experiments (fine-tuned pretrained backbones and everything built on them) are in git under the tag `pre-fresh-start`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu && pip install -e . --no-deps
pytest                                                 # under a minute on CPU, synthetic data
```

`requirements.txt` pins the exact versions in use. The extra index gives the CPU build of PyTorch; with a GPU, install the matching torch and torchvision builds first.

## Data

The label map, manifest, near-duplicate groups and splits are committed (`data/`). They are frozen: do not re-run the split step.

The images are not in the repository. Put them under `data/raw/` as laid out in [configs/data.yaml](configs/data.yaml):

| Dataset | Used for | Get it from |
| --- | --- | --- |
| Fitzpatrick17k | training, validation, test | github.com/mattgroh/fitzpatrick17k: `python -m skincancer.data.download_fitzpatrick17k` (dead links from the Wayback Machine, checked by MD5) |
| PAD-UFES-20 | training, validation, test | Mendeley Data; merge `imgs_part_*` into `images/` |
| DDI | a quarter training (since checkpoint 4), a quarter validation, half test ([DECISIONS.md](DECISIONS.md), section 26) | Stanford AIMI (request access); never shown in figures |
| DermaCon-IN | test only | Harvard Dataverse: `python -m skincancer.data.download_dermacon_in` |
| MRA-MIDAS | 40% training, 30% validation, 30% test, by patient (since checkpoint 2) | Stanford AIMI on Redivis: `python -m skincancer.data.download_midas` |
| MILK10k (clinical close-ups) | training only (since checkpoint 3; `data/extra/milk10k.csv`) | ISIC Archive, doi:10.34970/648456 |
| HIBA, MED-NODE, Derm7pt, Asan and Hallym, ISIC 2019, MCR-SL | test only, never trained on (`data/external.csv`, splits `ext_*`; section 27) | `python -m skincancer.data.extra --help`, then `python -m skincancer.data.external` |

The pipeline that built the committed files, for reference:

```bash
python -m skincancer.data.build_label_map   # data/label_map.csv
python -m skincancer.data.build_manifest    # data/manifest.csv
python -m skincancer.data.dedup             # data/groups.csv (near-duplicates + patients)
python -m skincancer.data.make_splits       # split column + data/splits/*.csv
python -m skincancer.data.count_check       # split × class and split × type-of-lesion tables
```

## Train and evaluate

```bash
python -m skincancer.train --config configs/skinnet.yaml --seed 0     # one seed
python -m skincancer.predict --run runs/skinnet/s0 --splits ddi_hold dermacon_test dermacon_other midas_test --final
python -m skincancer.evaluate --exp skinnet                           # results/skinnet/metrics.csv
python -m skincancer.analysis.gradcam --run runs/skinnet/s0 --split test
python -m skincancer.compare --a skinnet-c4 --b skinnet-c5 --splits val midas_val   # is b better than a?
python -m skincancer.plot_history --exp skinnet-c5                    # results/skinnet-c5/curves.png
```

For an idea under test, `scripts/run_idea.sh <letter>` trains `configs/skinnet-idea-<letter>.yaml`, averages the seeds and compares with the current checkpoint (`skinnet-c5`; set `REF`) on the validation sets only (`val`, `midas_val`, `ddi_val`); `python scripts/step1_gate.py runs/<idea>/logs/compare_val.log` applies the step 1 rule (section 26), and `bash scripts/step2.sh <idea>` scores an idea that passed it on every test-only set, once, and compares it with the checkpoint there. A kept idea becomes the next checkpoint with `python scripts/adopt_checkpoint.py <idea> skinnet-c<n> "<why>"`.

The queue does all of that for several ideas in a row, detached from the terminal, and logs to `runs/queue.log`. The letters below are examples:

```bash
bash scripts/queue.sh start ad ae             # skinnet-idea-ad, then -ae: step 1, its rule, step 2 if passed
bash scripts/queue.sh start ad@skinnet-c4     # against another reference (letter[@reference][:rule])
bash scripts/queue.sh start --wait ad         # start once the PanDerm repository's queue has finished
bash scripts/queue.sh status                  # which seed and epoch, time left, the last results
bash scripts/queue.sh stop                    # stop it now; the same start command carries on later
```

Finished seeds are skipped and a stopped seed resumes from its `last.pt`, so nothing is lost by stopping. The queue never adopts or deletes anything.

`--set key=value` overrides any config field, for example `--set model.width=48 --set name=skinnet_w48`.

## Layout

```
configs/data.yaml                 dataset paths and split settings
configs/                          skinnet.yaml (the recipe), checkpoints skinnet-c<n>.yaml, ideas skinnet-idea-<x>.yaml
docs/                             CHECKPOINTS (every kept model), IDEAS, RESULTS, DATA_PLAN, LABEL_MAP_REVIEW
tests/                            unit and pipeline tests (synthetic data, under a minute on CPU)
DECISIONS.md                      every rule and decision, in order
src/skincancer/models/            skinnet.py (SkinNet), __init__.py (the classifier head)
src/skincancer/train.py           training loop
src/skincancer/predict.py         scoring a finished run; the test-set ledger
src/skincancer/evaluate.py        metrics with confidence intervals
src/skincancer/compare.py         paired comparison of two models on the same images
src/skincancer/data/              label map, manifest, dedup, splits, outside sets, Derm1M, dataset and augmentation
scripts/                          queue.sh (ideas in a row) and status.py, run_idea.sh (step 1), step1_gate.py (its rule), step2.sh (test sets, once), adopt_checkpoint.py
runs/<name>/s<seed>/              best.pt  config.yaml  history.csv  summary.json  preds/   (local, not in git)
runs/final_ledger.csv             one row per (run, test-only set) ever scored
results/<name>/                   metrics.csv: one row per set x subgroup x metric (local)
checkpoints/<name>/               the kept models' weights (local)
```

Commands take just the model's name (`--exp skinnet-c5`); `predict --run` takes the full folder (`runs/skinnet-c5/s0`). Paths are in [paths.py](src/skincancer/paths.py).

`data/manifest.csv` has one row per image (cancer or not, type of lesion, dataset, group, split); [data/splits/SUMMARY.md](data/splits/SUMMARY.md) counts them. `runs/<name>/s<seed>/preds/<split>.csv` has the predicted class probabilities. Evaluation reads only those two.

## Rules the code enforces

- `val` picks the checkpoint. Test splits never choose anything.
- DDI's held-out half, DermaCon-IN, MIDAS test and the `ext_*` outside sets are never scored during training. `predict --final` scores them and refuses to score the same run on the same set twice.
- A near-duplicate group never spans two splits.
- Seeds 0, 1, 2. Results are the seed mean with a group-bootstrap 95% confidence interval.
