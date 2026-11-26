"""The queue's status report (scripts/status.py) on a made-up run folder."""
import importlib.util
from pathlib import Path

import pandas as pd
import yaml

spec = importlib.util.spec_from_file_location("status", Path(__file__).parents[1] / "scripts" / "status.py")
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


def _seed(root: Path, exp: str, seed: int, epochs_done: int, planned: int = 10, finished: bool = False):
    d = root / exp / f"s{seed}"
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump({"train": {"epochs": planned, "patience": 0}}), encoding="utf-8")
    pd.DataFrame({"epoch": range(epochs_done), "val_auc": [0.80 + 0.01 * e for e in range(epochs_done)],
                  "secs": [60.0] * epochs_done}).to_csv(d / "history.csv", index=False)
    if finished:
        (d / "summary.json").write_text('{"epochs_run": %d}' % epochs_done, encoding="utf-8")
    else:
        (d / "last.pt").write_bytes(b"")


def test_status_reports_the_seed_in_progress_and_what_is_left(tmp_path):
    _seed(tmp_path, "m-idea-a", 0, 10, finished=True)
    _seed(tmp_path, "m-idea-a", 1, 4)
    (tmp_path / "queue.log").write_text("[10:00] queue start: a\nnot a log line\n", encoding="utf-8")
    text = status.report(tmp_path)
    assert "m-idea-a s1: training, epoch 4 of 10, 60 s per epoch, best val 0.8300 at epoch 3" in text
    assert "at most 6 min left" in text and "then s2 (about 10 min)" in text
    assert "[10:00] queue start: a" in text and "not a log line" not in text
    assert "s1: stopped" in status.report(tmp_path, queue_running=False)


def test_status_with_nothing_running(tmp_path):
    assert status.report(tmp_path).startswith("no seed is in progress")


def test_adopt_checkpoint_renames_everything_and_relabels_tables(tmp_path):
    spec = importlib.util.spec_from_file_location("adopt", Path(__file__).parents[1] / "scripts" / "adopt_checkpoint.py")
    adopt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adopt)
    idea, name = "m-idea-q", "m-c1"
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / f"{idea}.yaml").write_text(f"# Idea Q: one change.\nname: {idea}\n", encoding="utf-8")
    run = tmp_path / "runs" / idea
    (run / "s0").mkdir(parents=True)
    (run / "s0" / "config.yaml").write_text(yaml.safe_dump({"name": idea}), encoding="utf-8")
    (run / "s0" / "best.pt").write_bytes(b"w")
    (run / "avg" / "preds").mkdir(parents=True)
    (run / "avg" / "config.yaml").write_text(yaml.safe_dump({"name": idea, "ensemble_of": [f"runs/{idea}/s0"]}), encoding="utf-8")
    pd.DataFrame({"image_id": ["a"], "exp": [idea], "p_cancer": [0.5]}).to_csv(run / "avg" / "preds" / "val.csv", index=False)
    (tmp_path / "results" / idea).mkdir(parents=True)
    pd.DataFrame({"exp": [idea], "metric": ["binary_auc"], "mean": [0.9]}).to_csv(tmp_path / "results" / idea / "metrics.csv", index=False)
    pd.DataFrame({"run": [f"runs/{idea}/avg", "runs/other/avg"], "split": ["test", "test"]}).to_csv(tmp_path / "runs" / "final_ledger.csv", index=False)

    adopt.main(idea, name, "Checkpoint 1 (test)", root=tmp_path)

    assert not (tmp_path / "configs" / f"{idea}.yaml").exists() and not run.exists()
    assert "name: m-c1" in (tmp_path / "configs" / f"{name}.yaml").read_text(encoding="utf-8")
    new = tmp_path / "runs" / name
    assert yaml.safe_load((new / "avg" / "config.yaml").read_text(encoding="utf-8"))["ensemble_of"] == [f"runs/{name}/s0"]
    assert pd.read_csv(new / "avg" / "preds" / "val.csv")["exp"].tolist() == [name]
    assert pd.read_csv(tmp_path / "results" / name / "metrics.csv")["exp"].tolist() == [name]
    assert pd.read_csv(tmp_path / "runs" / "final_ledger.csv")["run"].tolist() == [f"runs/{name}/avg", "runs/other/avg"]
    ck = tmp_path / "checkpoints" / name
    assert (ck / "s0" / "best.pt").exists() and (ck / "metrics.csv").exists()
    assert pd.read_csv(ck / "avg" / "preds" / "val.csv")["exp"].tolist() == [name]
