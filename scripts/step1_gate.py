"""Step 1 verdict from a compare log (DECISIONS.md, sections 26, 29 and 30). Exit 0 = passed.

    python scripts/step1_gate.py runs/<idea>/logs/compare_val.log gain
    gain:  higher on all three of val (within-source AUC), midas_val and ddi_val, and lower on none
           by a supported amount (section 26's rule; the default for an idea)
    guard: only "lower on none by a supported amount" (section 29)
"""
import sys

import pandas as pd


def verdict(log: str, mode: str = "gain") -> tuple[bool, str]:
    t = pd.read_csv(log, sep=r"\s+", skiprows=1)                 # the table compare.py prints
    rows = {("val", "source_auc"), ("midas_val", "auc"), ("ddi_val", "auc")}
    t = t[[(s, m) in rows for s, m in zip(t["split"], t["metric"])]]
    if len(t) != 3:
        raise ValueError(f"{log}: expected val, midas_val and ddi_val, found {len(t)} rows")
    lower = ((t["supported"] == "yes") & (t["diff"] < 0)).any()
    ok = bool(not lower and (mode == "guard" or (t["diff"] > 0).all()))
    detail = ", ".join(f"{s} {d:+.4f}{'*' if sp == 'yes' else ''}"
                       for s, d, sp in zip(t["split"], t["diff"], t["supported"]))
    return ok, f"{'PASSED' if ok else 'FAILED'} ({mode}): {detail}"


if __name__ == "__main__":
    ok, text = verdict(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "gain")
    print(text)
    sys.exit(0 if ok else 1)
