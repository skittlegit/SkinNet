"""Make a kept idea the next checkpoint: one name everywhere (DECISIONS.md, section 15).

    python scripts/adopt_checkpoint.py skinnet-idea-y skinnet-c6 "Checkpoint 6 (DECISIONS.md, section 33): was idea Y"

Renames, in place:
- configs/<track>/<idea>.yaml -> <name>.yaml, with `name:` and the first comment line updated;
- runs/<track>/<idea>/ -> runs/<track>/<name>/, and `name:` in every config.yaml inside;
- results/<track>/<idea>/ -> results/<track>/<name>/ (if it exists);
- every runs/final_ledger.csv row that names the idea's folders.
Then copies each seed's best.pt, config.yaml and summary.json, and the seed average's config and
predictions, to checkpoints/<track>/<name>/. Prints what it did; refuses if <name> exists anywhere.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skincancer.tracks import TRACK  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(idea: str, name: str, header: str) -> None:
    cfg_old, cfg_new = ROOT / "configs" / TRACK / f"{idea}.yaml", ROOT / "configs" / TRACK / f"{name}.yaml"
    run_old, run_new = ROOT / "runs" / TRACK / idea, ROOT / "runs" / TRACK / name
    res_old, res_new = ROOT / "results" / TRACK / idea, ROOT / "results" / TRACK / name
    ckpt = ROOT / "checkpoints" / TRACK / name
    for p in (cfg_new, run_new, res_new, ckpt):
        if p.exists():
            raise SystemExit(f"{p} already exists")
    if not (cfg_old.exists() and run_old.exists()):
        raise SystemExit(f"{idea}: config or runs missing")

    text = cfg_old.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^name: .*$", f"name: {name}", text, count=1)
    lines = text.splitlines()
    old_head = []
    while lines and lines[0].startswith("#"):   # the idea's header comment becomes the checkpoint's second line
        old_head.append(lines.pop(0))
    head = [f"# {header}."] + [("# Was: " + h.lstrip("# ")) if i == 0 else h for i, h in enumerate(old_head)]
    cfg_new.write_text("\n".join(head + lines) + "\n", encoding="utf-8", newline="\n")
    cfg_old.unlink()
    print(f"config  {cfg_old.name} -> {cfg_new.name}")

    run_old.rename(run_new)
    for c in run_new.glob("*/config.yaml"):
        d = yaml.safe_load(c.read_text(encoding="utf-8"))
        d["name"] = name
        c.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    print(f"runs    {run_old.relative_to(ROOT)} -> {run_new.relative_to(ROOT)}")
    if res_old.exists():
        res_old.rename(res_new)
        print(f"results {res_old.relative_to(ROOT)} -> {res_new.relative_to(ROOT)}")

    ledger = ROOT / "runs" / "final_ledger.csv"
    if ledger.exists():
        df = pd.read_csv(ledger)
        old_prefix, new_prefix = f"runs/{TRACK}/{idea}/", f"runs/{TRACK}/{name}/"
        hit = df["run"].astype(str).str.startswith(old_prefix)
        df.loc[hit, "run"] = df.loc[hit, "run"].str.replace(old_prefix, new_prefix, regex=False)
        df.to_csv(ledger, index=False, lineterminator="\n")
        print(f"ledger  {int(hit.sum())} rows renamed")

    for sdir in sorted(p for p in run_new.iterdir() if p.is_dir() and (p.name.startswith("s") or p.name == "avg")):
        dst = ckpt / sdir.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("best.pt", "config.yaml", "summary.json"):
            if (sdir / f).exists():
                shutil.copy2(sdir / f, dst / f)
        if sdir.name == "avg" and (sdir / "preds").exists():
            shutil.copytree(sdir / "preds", dst / "preds")
    print(f"weights copied to {ckpt.relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
