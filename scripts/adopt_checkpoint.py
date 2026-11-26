"""Make a kept idea the next checkpoint: one name everywhere (DECISIONS.md, section 15).

    python scripts/adopt_checkpoint.py skinnet-idea-ad skinnet-c6 "Checkpoint 6 (DECISIONS.md, section NN): was idea AD"

Renames, in place:
- configs/<idea>.yaml -> <name>.yaml, with `name:` and the first comment line updated;
- runs/<idea>/ -> runs/<name>/, and `name:` in every config.yaml inside;
- results/<idea>/ -> results/<name>/ (if it exists);
- every runs/final_ledger.csv row that names the idea's folders;
- the `exp` label in its prediction and score tables, and the seed average's `ensemble_of` list.
Then copies each seed's best.pt, config.yaml and summary.json, the seed average's config and
predictions, and its score table, to checkpoints/<name>/. Prints what it did; refuses if <name> exists anywhere.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def relabel(folder: Path, idea: str, name: str) -> int:
    """Every table under `folder` whose `exp` column names the idea now names the checkpoint."""
    n = 0
    for f in folder.rglob("*.csv"):
        df = pd.read_csv(f)
        if "exp" in df.columns and (df["exp"] == idea).any():
            df.loc[df["exp"] == idea, "exp"] = name
            df.to_csv(f, index=False, lineterminator="\n")
            n += 1
    return n


def main(idea: str, name: str, header: str, root: Path = ROOT) -> None:
    cfg_old, cfg_new = root / "configs" / f"{idea}.yaml", root / "configs" / f"{name}.yaml"
    run_old, run_new = root / "runs" / idea, root / "runs" / name
    res_old, res_new = root / "results" / idea, root / "results" / name
    ckpt = root / "checkpoints" / name
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
        if d.get("ensemble_of"):   # the seed average lists its seeds' folders
            d["ensemble_of"] = [s.replace(f"runs/{idea}/", f"runs/{name}/") for s in d["ensemble_of"]]
        c.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    print(f"runs    {run_old.relative_to(root)} -> {run_new.relative_to(root)}")
    if res_old.exists():
        res_old.rename(res_new)
        print(f"results {res_old.relative_to(root)} -> {res_new.relative_to(root)}")
    print(f"labels  {relabel(run_new, idea, name) + (relabel(res_new, idea, name) if res_new.exists() else 0)} tables renamed")

    ledger = root / "runs" / "final_ledger.csv"
    if ledger.exists():
        df = pd.read_csv(ledger)
        old_prefix, new_prefix = f"runs/{idea}/", f"runs/{name}/"
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
    if (res_new / "metrics.csv").exists():
        shutil.copy2(res_new / "metrics.csv", ckpt / "metrics.csv")
    print(f"weights copied to {ckpt.relative_to(root)}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
