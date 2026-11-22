"""Outside test sets that no model trains on (DECISIONS.md, section 27).

    python -m skincancer.data.external            # build data/external.csv

Each set becomes a test-only split `ext_<name>`, kept in data/external.csv beside the frozen
manifest (runs.read_manifest joins the two). A photo that near-duplicates any training image
(manifest train, midas_train, ddi_train, ddi_val, and MILK10k) is excluded, by the same
perceptual-hash and embedding checks as the manifest (configs/data.yaml, dedup).

Sets: HIBA (Argentina, clinical photos, biopsied), MED-NODE (Groningen), MCR-SL (Norway; when
downloaded), Derm7pt clinical photos, Asan and Hallym (Korea; 99 px thumbnails, biopsied only),
and the ISIC 2019 subsample already on disk (dermoscopy: a different kind of image).
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_data_config
from ..labels import CANCER_TYPES, TUMOUR, map_diagnosis
from .dedup import cosine_pairs, embed, hash_pairs, phash_ints
from .extra import CLINICAL_TYPES, asan_rows, derm7pt_rows, isic2019_rows, isic_binary, load_extra

OUT = Path("data/external.csv")
TRAIN_SPLITS = ["train", "midas_train", "ddi_train", "ddi_val"]
COLS = ["image_id", "group_id", "dataset", "path", "diagnosis", "label_4class", "label_fine", "label_binary",
        "label_cancer", "fst", "fst_bucket", "site", "is_acral", "patient_id"]


def _bucket(f) -> float | str:
    return {"I": "I-III", "II": "I-III", "III": "I-III", "IV": "IV", "V": "V-VI", "VI": "V-VI"}.get(f, np.nan)


def hiba_rows(root: Path) -> pd.DataFrame:
    """HIBA clinical photos (ISIC collection 175, CC BY). Labels as MILK10k: biopsy needed."""
    rows = [r for r in json.loads((root / "hiba" / "metadata.json").read_text(encoding="utf-8"))
            if r["metadata"]["acquisition"].get("image_type") in CLINICAL_TYPES]
    clin = pd.DataFrame([r["metadata"]["clinical"] for r in rows])
    specific, label, binary = isic_binary(clin, need_biopsy=True)
    df = pd.DataFrame({
        "image_id": ["hiba_" + r["isic_id"] for r in rows],
        "group_id": "hiba_" + clin["lesion_id"].astype(str), "dataset": "hiba",
        "path": ["hiba/images/" + r["isic_id"] + ".jpg" for r in rows],
        "diagnosis": specific, "label_4class": label,
        "label_fine": label.where(label.isin(TUMOUR), np.where(binary == 0.0, "other_benign_tumour", "unlabelled")),
    })
    df["label_binary"] = binary
    df["label_cancer"] = df["label_binary"].map({1.0: "cancer", 0.0: "not_cancer"})
    df["fst"] = clin.get("fitzpatrick_skin_type")
    df["fst_bucket"] = df["fst"].map(_bucket)
    df["site"] = clin.get("anatom_site_1")
    df["is_acral"] = df["site"].astype(str).str.contains("acral|palm|sole", case=False)
    df["patient_id"] = "hiba_" + clin["patient_id"].astype(str)
    return df


def mednode_rows(root: Path) -> pd.DataFrame:
    """MED-NODE: 70 melanomas, 100 nevi (clinical photos, University Medical Center Groningen)."""
    d = root / "mednode"
    if not (d / "complete_mednode_dataset").exists():
        with zipfile.ZipFile(d / "complete_mednode_dataset.zip") as z:
            for n in z.namelist():                       # names checked: no absolute or parent paths
                if n.startswith("/") or ".." in n:
                    raise ValueError(f"unsafe path in zip: {n}")
            z.extractall(d)
    recs = []
    for cls, diag in (("melanoma", "melanoma"), ("naevus", "nevus")):
        for p in sorted((d / "complete_mednode_dataset" / cls).glob("*.jpg")):
            recs.append({"image_id": f"mednode_{cls}_{p.stem}", "path": p.relative_to(root).as_posix(), "diagnosis": diag})
    df = pd.DataFrame(recs)
    df["group_id"] = df["image_id"]
    df["dataset"] = "mednode"
    df["label_4class"] = df["diagnosis"].map(lambda s: map_diagnosis(s)[0])
    df["label_fine"] = df["label_4class"]
    df["label_binary"] = df["label_4class"].map(lambda l: 1.0 if l in CANCER_TYPES else 0.0)
    df["label_cancer"] = df["label_binary"].map({1.0: "cancer", 0.0: "not_cancer"})
    df["fst"] = np.nan
    df["fst_bucket"] = np.nan
    df["site"] = np.nan
    df["is_acral"] = False
    df["patient_id"] = df["group_id"]
    return df


def _from_extra(fn, root: Path, name: str, subsample: bool) -> pd.DataFrame:
    df = fn(root)
    if subsample and "in_subsample" in df:
        df = df[df["in_subsample"]]
    df = df.drop(columns=[c for c in ("in_subsample", "licence") if c in df])
    df["dataset"] = name
    return df


def builders(root: Path) -> dict:
    out = {
        "hiba": lambda: hiba_rows(root),
        "mednode": lambda: mednode_rows(root),
        "derm7pt": lambda: _from_extra(derm7pt_rows, root, "derm7pt", False),
        "asan": lambda: _from_extra(asan_rows, root, "asan", False),
        "isic2019": lambda: _from_extra(isic2019_rows, root, "isic2019", True),   # the 5,000 on disk
    }
    return {k: v for k, v in out.items() if (root / {"derm7pt": "derm7pt", "asan": "asan_hallym"}.get(k, k)).exists()}


def build(cfg: dict) -> pd.DataFrame:
    root = Path(cfg["data_root"])
    parts = []
    for name, fn in builders(root).items():
        df = fn()
        df = df[df["label_binary"].notna() & df["path"].map(lambda p: (root / p).exists())]
        print(f"{name}: {len(df)} labelled photos ({int(df['label_binary'].sum())} cancers)", flush=True)
        parts.append(df.reindex(columns=COLS))
    ext = pd.concat(parts, ignore_index=True)
    manifest = pd.read_csv(cfg["manifest"], low_memory=False)
    train = pd.concat([manifest[manifest["split"].isin(TRAIN_SPLITS)][["path"]],
                       load_extra(["milk10k"], "label_cancer", ["not_cancer", "cancer"])[["path"]]], ignore_index=True)
    paths = [root / p for p in ext["path"]] + [root / p for p in train["path"]]
    n = len(ext)
    dup = np.zeros(n, bool)
    for i, j in hash_pairs(phash_ints(paths), cfg["dedup"]["phash_max_distance"]):
        if i < n <= j:
            dup[i] = True
    for i, j in cosine_pairs(embed(paths), cfg["dedup"]["embed_min_cosine"]):
        if i < n <= j:
            dup[i] = True
    ext["exclude_reason"] = np.where(dup, "dup_of_train", None)
    ext["split"] = np.where(dup, "excluded", "ext_" + ext["dataset"])
    print(f"{dup.sum()} of {n} outside photos near-duplicate one of {len(train)} training images: excluded")
    return ext


def main(argv=None):
    cfg = load_data_config("configs/data.yaml")
    ext = build(cfg)
    ext.to_csv(OUT, index=False)
    print(pd.crosstab(ext["split"], ext["label_cancer"], margins=True).to_string())


if __name__ == "__main__":
    main()
