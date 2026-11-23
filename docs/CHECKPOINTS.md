# Checkpoints

Each track's current model is below; the earlier checkpoints and what was tried on the way are in [SkinNet checkpoints](skinnet/CHECKPOINTS.md) and [PanDerm checkpoints](panderm/CHECKPOINTS.md). Every idea by letter, kept and discarded: [SkinNet ideas](skinnet/IDEAS.md), [PanDerm ideas](panderm/IDEAS.md).

A checkpoint is a model we have decided to keep and build on. Each new idea starts from the latest checkpoint and has to beat it to become the next one. A checkpoint also records everything that was tried on the way to it, so the same experiments are not run twice.

Scores are the mean of three seeds. Details and confidence intervals: [SkinNet results](skinnet/RESULTS.md), [PanDerm results](panderm/RESULTS.md), and both side by side in [COMPARISON.md](COMPARISON.md).

## Names

| Model | Name (folder in `runs/<track>/`, `results/<track>/`) | Built from |
| --- | --- | --- |
| Baseline | `skinnet` | `configs/skinnet/skinnet.yaml` |
| Checkpoint 1 | `skinnet-c1` | `configs/skinnet/skinnet-c1.yaml` |
| Checkpoint 2 | `skinnet-c2` | `skinnet-c1`'s three seeds, predicting with test-time flips, and their average |
| Checkpoint 3 | `skinnet-c3` | `configs/skinnet/skinnet-c3.yaml` |
| Checkpoint 4 | `skinnet-c4` | `configs/skinnet/skinnet-c4.yaml` (was idea Q, `skinnet-idea-q`) |
| PanDerm checkpoint 1 | `panderm-c1` | `configs/panderm/panderm-c1.yaml` (PanDerm track, idea PC; DECISIONS.md, section 24) |
| Checkpoint 5 | `skinnet-c5` | `configs/skinnet/skinnet-c5.yaml` (was idea U, `skinnet-idea-u`; DECISIONS.md, section 26) |
| PanDerm checkpoint 2 | `panderm-c2` | `configs/panderm/panderm-c2.yaml` (was idea PH, `panderm-idea-h`; section 26) |
| Checkpoint 5, all-DDI version | `skinnet-idea-t` | `configs/skinnet/skinnet-idea-t.yaml`: checkpoint 5 with `ddi_val` also in training (section 26) |
| PanDerm checkpoint 2, all-DDI version | `panderm-idea-g` | `configs/panderm/panderm-idea-g.yaml`: PanDerm checkpoint 2 with `ddi_val` also in training (section 26) |

One name per model, the same for its config, run folder and result folder. A run folder `runs/<track>/<name>/` holds the seeds (`s0`, `s1`, `s2`) and their average (`avg`); `results/<track>/<name>/metrics.csv` is the model's full score table. The track is `panderm` for names starting `panderm` and `skinnet` for every other. An idea being tested is `skinnet-idea-<letter>`; if kept it is renamed to the next checkpoint (`skinnet-c6`, `panderm-c3`), and if discarded it is deleted. The rule is in [DECISIONS.md](../DECISIONS.md), section 15.

## Checkpoint 5 and PanDerm checkpoint 2 (22 Nov 2026): each track's current model

Each is its track's previous checkpoint with a quarter of DDI (`ddi_train`, 164 photos) added to training. Another quarter (`ddi_val`, 164) is a validation set, and the other half (`ddi_hold`, 328) is the test (DECISIONS.md, section 26). The user chose this on 22 Nov for the DDI validation set, which every later idea can now be checked on.

DDI is now partly seen in training. `ddi_hold` holds other photos from DDI's hospital, so it measures how well a model learned that hospital, not how well it carries over to a new one. The never-seen DDI scores remain those of checkpoint 4 and PanDerm checkpoint 1.

| | Checkpoint 5 | PanDerm checkpoint 2 |
| --- | --- | --- |
| Name | `skinnet-c5` (was idea U) | `panderm-c2` (was idea PH) |
| Training data | 18,186 images: checkpoint 4's 18,022 + 164 of `ddi_train` | The same 18,186 |
| Config | `configs/skinnet/skinnet-c5.yaml` | `configs/panderm/panderm-c2.yaml` |
| Run folder | `runs/skinnet/skinnet-c5` | `runs/panderm/panderm-c2` |
| Weights | `checkpoints/skinnet/skinnet-c5/` (local, git-ignored) | `checkpoints/panderm/panderm-c2/` (local, git-ignored) |
| Git tag | `checkpoint-5` | `panderm-checkpoint-2` |

Scores are AUC; `val` is within-source AUC. Bold marks a supported difference.

| Set | Checkpoint 4 | Checkpoint 5 | Difference [95% interval] | PanDerm c1 | PanDerm c2 | Difference [95% interval] |
| --- | --- | --- | --- | --- | --- | --- |
| `val` | 0.905 | 0.903 | −0.003 [−0.012, +0.006] | 0.983 | 0.980 | −0.003 [−0.007, +0.001] |
| `midas_val` | 0.818 | 0.808 | −0.010 [−0.030, +0.010] | 0.856 | 0.870 | +0.013 [−0.003, +0.030] |
| `ddi_val` | 0.701 | 0.749 | **+0.048 [+0.015, +0.087]** | 0.677 | 0.764 | **+0.086 [+0.033, +0.147]** |
| `test` | 0.903 | 0.901 | −0.002 [−0.007, +0.003] | 0.977 | 0.976 | −0.001 [−0.006, +0.004] |
| `midas_test` | 0.750 | 0.744 | −0.006 [−0.026, +0.012] | 0.874 | 0.874 | −0.001 [−0.017, +0.017] |
| `ddi_hold` | 0.641 | 0.670 | **+0.029 [+0.007, +0.052]** | 0.691 | 0.733 | **+0.042 [+0.003, +0.082]** |
| DermaCon-IN (65 photos) | 0.818 | 0.847 | +0.029 [−0.014, +0.084] | 0.850 | 0.882 | +0.032 [−0.006, +0.104] |

**The all-DDI versions** (ideas T and PG). They are each checkpoint's recipe with `ddi_val` added to training (328 DDI photos), kept by the user on 22 Nov as the figures reported for each track for now. New ideas are still compared with checkpoint 5 and PanDerm checkpoint 2. Weights are in `checkpoints/skinnet/skinnet-idea-t/` and `checkpoints/panderm/panderm-idea-g/`; git tags `checkpoint-5-all-ddi` and `panderm-checkpoint-2-all-ddi`. Their `ddi_hold` scores are a little optimistic, since `ddi_hold` chose the design. On `ddi_hold`:
- T scored 0.684: checkpoint 5 − T = −0.014, not supported.
- PG scored 0.785: PanDerm checkpoint 2 − PG = −0.052 [−0.088, −0.017], supported.

**Across tracks** (`panderm-c2` − `skinnet-c5`), all supported except DermaCon-IN:
- `test` +0.075
- `midas_test` +0.130
- `ddi_hold` +0.063 [+0.006, +0.118]
- DermaCon-IN +0.036, not supported

**Where they stand.** Neither passed step 1 as written, and neither rose on every test set (DECISIONS.md, section 19). They are checkpoints by the user's decision, as checkpoint 4 was by the team's. On the PanDerm track the choice costs 0.052 on `ddi_hold` against PG, a supported amount, in exchange for a DDI validation set.

## How to use a checkpoint

```bash
# checkpoint 1, one seed (weights are local, not in git)
python -m skincancer.predict --run checkpoints/skinnet/skinnet-c1/s0 --splits val

# checkpoint 2: rebuild it from checkpoint 1's runs
python -m skincancer.derive tta --exp skinnet-c1 --name skinnet-c2
for s in 0 1 2; do python -m skincancer.predict --run runs/skinnet/skinnet-c2/s$s --splits val test; done
python -m skincancer.derive ensemble --exp skinnet-c2 --splits val test
python -m skincancer.compare --a skinnet-c1 --b skinnet-c2 --splits val test

# checkpoint 3: train it (flips are in its config), then average the seeds
for s in 0 1 2; do python -m skincancer.train --config configs/skinnet/skinnet-c3.yaml --seed $s; done
python -m skincancer.derive ensemble --exp skinnet-c3 --splits val test midas_val

# return the code to the state that produced a checkpoint
git checkout checkpoint-3
```

The weights are not in the repository (`checkpoints/` is git-ignored). Retraining from the tagged code with seeds 0, 1 and 2 reproduces them.
