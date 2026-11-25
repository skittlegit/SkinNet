# SkinNet: skin cancer detection

Skin cancer detection from clinical skin photos: the model tells whether a photo shows a skin cancer. SkinNet is a small residual CNN of our own, trained from random weights on our data: no pretrained weights, no frozen layer.

It is one of two tracks of one project. The other, **PanDerm** (PanDerm's ViT, pretrained on about 2.1 million skin images and fine-tuned on our data), lives in its own repository, `skittlegit/PanDerm`. The two shared this repository until 24 Nov 2026 (DECISIONS.md, section 31), so its history holds both. Both tracks use the same data, splits and rules.

The models train on about 18,350 photos and are tested on held-back photos, on MIDAS, DDI and DermaCon-IN (Indian), and on six outside sets no model ever trains on (section 27). This is stage one. Telling cancer types apart, and whether darker skin needs its own model, come later.

This is a research prototype, not a diagnostic tool.

**Read [DECISIONS.md](DECISIONS.md) before changing anything.** It holds the rules for labels, splits, the network, the training recipe and the metrics, and the code enforces them. Sections 1 to 31 are shared with PanDerm; from section 32 on, each repository keeps its own.

The project was restarted on 12 Nov 2026. The earlier experiments (fine-tuned pretrained backbones and everything built on them) are in git under the tag `pre-fresh-start`.

Where to read:
- The current model: [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) (as of the split, both tracks).
- SkinNet against PanDerm (outside sets, leave-one-dataset-out, DDI), as of the split: [docs/COMPARISON.md](docs/COMPARISON.md).
- SkinNet: [results](docs/skinnet/RESULTS.md), [ideas](docs/skinnet/IDEAS.md), [earlier checkpoints](docs/skinnet/CHECKPOINTS.md).
- Data sources, tried and planned: [docs/DATA_PLAN.md](docs/DATA_PLAN.md).

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
| DDI | test only for the checkpoints; ideas T, U, PG, PH (section 26) train on part of it | Stanford AIMI (request access); never shown in figures |
| DermaCon-IN | test only | Harvard Dataverse: `python -m skincancer.data.download_dermacon_in` |
| MRA-MIDAS | 40% training, 30% validation, 30% test, by patient (since checkpoint 3) | Stanford AIMI on Redivis: `python -m skincancer.data.download_midas` |
| MILK10k (clinical close-ups) | training only (since checkpoint 4; `data/extra/milk10k.csv`) | ISIC Archive, doi:10.34970/648456 |
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
python -m skincancer.train --config configs/skinnet/skinnet.yaml --seed 0     # one seed
python -m skincancer.predict --run runs/skinnet/skinnet/s0 --splits ddi_test dermacon_test dermacon_other midas_test --final
python -m skincancer.evaluate --exp skinnet                           # results/skinnet/skinnet/metrics.csv
python -m skincancer.analysis.gradcam --run runs/skinnet/skinnet/s0 --split test
python -m skincancer.compare --a skinnet-c1 --b skinnet-c2 --splits val test   # is b better than a?
python -m skincancer.plot_history --exp skinnet-c6                    # results/skinnet/skinnet-c6/curves.png
```

Or all three seeds, the test-only sets and the metrics in one go:

```bash
PY=.venv/Scripts/python bash scripts/run.sh        # Windows venv; PY=python elsewhere
```

For an idea under test, `scripts/run_idea.sh <letter>` trains `configs/skinnet/skinnet-idea-<letter>.yaml`, averages the seeds and compares with the current checkpoint (`skinnet-c6`; set `REF`) on the validation sets only (`val`, `midas_val`, `ddi_val`); `python scripts/step1_gate.py runs/skinnet/<idea>/logs/compare_val.log` applies the step 1 rule (section 26), and `bash scripts/step2.sh <idea>` scores an idea that passed it on every test-only set, once, and compares it with the checkpoint there.

`--set key=value` overrides any config field, for example `--set model.width=48 --set name=skinnet_w48`.

## Layout

```
configs/data.yaml                 dataset paths and split settings
configs/skinnet/                  skinnet.yaml (the recipe), checkpoints skinnet-c<n>.yaml, ideas, leave-one-out
docs/CHECKPOINTS.md               the current models (both tracks, as of the split)
docs/COMPARISON.md                SkinNet and PanDerm side by side, as of the split
docs/skinnet/                     RESULTS, IDEAS and earlier CHECKPOINTS
docs/DATA_PLAN.md                 data sources
DECISIONS.md                      every rule and decision, in order
src/skincancer/models/            skinnet.py (SkinNet), __init__.py (the classifier head)
src/skincancer/train.py           training loop
src/skincancer/predict.py         scoring a finished run; the test-set ledger
src/skincancer/evaluate.py        metrics with confidence intervals
src/skincancer/compare.py         paired comparison of two models on the same images
src/skincancer/data/              label map, manifest, dedup, splits, outside sets, Derm1M, dataset and augmentation
scripts/                          run.sh, run_idea.sh (step 1), step1_gate.py (the step 1 rule), step2.sh (test sets, once), run_reference.sh
runs/skinnet/<name>/s<seed>/      best.pt  config.yaml  history.csv  summary.json  preds/   (local, not in git)
runs/final_ledger.csv             one row per (run, test-only set) ever scored (both tracks before the split)
results/skinnet/<name>/           metrics.csv: one row per set x subgroup x metric (local)
checkpoints/skinnet/              the kept models' weights (local)
```

Commands take just the model's name (`--exp skinnet-c6`); `predict --run` takes the full folder (`runs/skinnet/skinnet-c6/s0`). The `skinnet/` level in `runs/`, `results/` and `checkpoints/` is kept from before the split ([tracks.py](src/skincancer/tracks.py)), so every path in the ledger and the docs still holds.

`data/manifest.csv` has one row per image (cancer or not, type of lesion, dataset, group, split); [data/splits/SUMMARY.md](data/splits/SUMMARY.md) counts them. `runs/skinnet/<name>/s<seed>/preds/<split>.csv` has the predicted class probabilities. Evaluation reads only those two.

## Rules the code enforces

- `val` picks the checkpoint. Test splits never choose anything.
- DDI's held-out half, DermaCon-IN, MIDAS test and the `ext_*` outside sets are never scored during training. `predict --final` scores them and refuses to score the same run on the same set twice.
- A near-duplicate group never spans two splits.
- Seeds 0, 1, 2. Results are the seed mean with a group-bootstrap 95% confidence interval.
