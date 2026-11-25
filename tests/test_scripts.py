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
