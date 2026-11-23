"""Derm1M mycosis fungoides photos and look-alike rashes from the same sources, prepared for a
possible training idea (no training here; the user decides). Derm1M: Yan et al. 2025,
huggingface.co/datasets/redlessone/Derm1M, research use only; images never go in git.

    python -m skincancer.data.derm1m                 # -> data/extra/derm1m_mf.csv, images extracted
    python -m skincancer.data.derm1m --sheet         # + contact sheets in reports/ (git-ignored)
    python -m skincancer.data.derm1m --review-sheets # numbered sheets of the chosen photos not yet reviewed

Steps:
1. Labels. Derm1M's labels come from captions, not biopsies. Kept: rows whose label is mycosis
   fungoides, cutaneous T-cell lymphoma or Sezary syndrome and nothing else (lists such as
   "eczema, mycosis fungoides, psoriasis" are differentials, left out), and single-label rows of
   rashes that mycosis fungoides is mistaken for. A plain "mycosis" search would also catch
   onychomycosis and other fungal infections.
2. Pictures that are not clinical photos of skin: captions naming histology, staining, microscopy,
   dermoscopy, scans or charts are left out, and so are photos of the mouth, tongue, gums, lips,
   eyes, nails, genital mucosa or endoscopy (caption or Derm1M's body location); then images
   under 128 px or longer than 2.5 to 1 (strips of panels).
3. Source groups: one YouTube video, one journal figure, one book page or one forum post is one
   group, at most 5 photos from each, so one video cannot fill the set.
4. Shortcuts. If every cancer came from YouTube and journals and nothing benign did, a model
   could learn the source; if the rashes were on darker skin than the mycosis fungoides photos,
   it could learn skin tone. Up to POOL times as many rashes as needed are drawn per source,
   each photo gets a rough skin tone (ITA of its skin-coloured pixels: light > 28, medium 5-28,
   dark <= 5; on Fitzpatrick17k it ranks with the dermatologists' types at Spearman -0.58), and
   the rashes are chosen to the same number per source and tone as the mycosis fungoides photos.
   Rashes not chosen are split `derm1m_spare` and never used.
5. A look at every chosen photo (data/derm1m_review.csv, one verdict per Derm1M file): not_skin,
   no_lesion and unusable photos are left out, and the next rash in line takes a dropped rash's
   place. Each pass of --review-sheets shows only the chosen photos without a verdict.
6. Near-duplicates (configs/data.yaml, dedup): a photo matching any validation or test image,
   the manifest's or the outside sets', is dropped, and so is one matching a training image (the
   manifest's `train` or MILK10k), which would flatter derm1m_test; near-duplicates inside Derm1M
   join one group.
7. Split by group: 70% `train_extra`, 30% `derm1m_test`, by label, so a held-back part of
   Derm1M can judge the idea.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from ..config import load_data_config
from .dedup import cosine_pairs, embed, hash_pairs, phash_ints
from .extra import EXTRA_DIR, EXTRA_SPLIT, NOT_EVALUATION

RAW = Path("derm1m")                     # under data_root
OUT = EXTRA_DIR / "derm1m_mf.csv"
TEST_SPLIT = "derm1m_test"
MF = re.compile(r"^(mycosis fungoides|cutaneous t[- ]cell lymphoma|s.zary syndrome)"
                r"(?: \(mf\)| manifestation| with erythroderma)?$")
MIMICS = ["psoriasis", "eczema", "lichen planus", "contact dermatitis", "tinea", "pityriasis rosea",
          "atopic dermatitis", "seborrheic dermatitis", "tinea corporis", "nummular eczema"]
NOT_A_PHOTO = re.compile(r"histolog|histopath|h&e|haematoxylin|hematoxylin|microscop|immunohisto|immunostain|stain"
                         r"|magnification|dermoscop|cytometr|\bpet[- ]ct\b|\bmri\b|x-ray|radiograph|diagram|chart"
                         r"|graph|flowchart|survival|kaplan", re.I)
NOT_SKIN = re.compile(r"\boral\b|\bmouth|tongue|gingiv|\bgums?\b|buccal|mucos|palat|endoscop|colonoscop|gastr|oesophag|esophag"
                      r"|intestin|bowel|conjunctiv|\bnails?\b|onycho|vulva|vagin|glans", re.I)
NOT_SKIN_SITE = {"oral", "mouth", "tongue", "gums", "lips", "eye", "eyes", "nails", "nail"}
PER_GROUP = 5
POOL = 3                                  # rashes drawn per mycosis fungoides photo, before matching
TONE_CUTS = (5.0, 28.0)                   # ITA: dark <= 5 < medium <= 28 < light
SPARE_SPLIT = "derm1m_spare"
REVIEW = Path("data/derm1m_review.csv")
DROP_VERDICTS = {"not_skin", "no_lesion", "unusable"}
TRAINED_EXTRAS = ["milk10k"]           # extra sets the current checkpoints train on (configs/skinnet/skinnet-c4.yaml)
MIN_SIDE, MAX_ASPECT = 128, 2.5
ZIP_OF = {"IIYI": "IIYI.zip", "edu": "edu.zip", "note": "note.zip", "public": "public.zip", "pubmed": "pubmed.zip",
          "reddit": "reddit.zip", "twitter": "twitter.zip", "youtube": "youtube.zip"}


def source_group(filename: str) -> str:
    folder, name = filename.split("/", 1)
    stem = Path(name).stem
    if folder == "youtube":
        return "yt_" + stem[:11]                       # YouTube ids are 11 characters
    return folder + "_" + re.sub(r"_\d+$", "", stem)   # figure, page or post: drop the panel number


def skin_site(meta: pd.DataFrame) -> pd.Series:
    """False for photos of the mouth, lips, eyes, nails, mucosa or an endoscopy, by caption or body location."""
    sites = meta["body_location"].astype(str).str.lower().str.split(r",\s*")
    return ~(meta["caption"].astype(str).str.contains(NOT_SKIN) | sites.map(lambda xs: bool(NOT_SKIN_SITE & set(xs))))


def select(meta: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Every mycosis fungoides photo (at most PER_GROUP per source group) and a pool of up to POOL
    times as many rashes per source, in a fixed random order (`rank`) that matching follows."""
    lab = meta["disease_label"].astype(str).str.lower().str.strip()
    single = ~lab.str.contains(",")
    photo = ~meta["caption"].astype(str).str.contains(NOT_A_PHOTO) & skin_site(meta)
    df = meta.assign(label=lab, group=meta["filename"].map(source_group))
    mf = df[single & lab.str.match(MF) & photo].assign(label_binary=1.0)
    mim = df[single & lab.isin(MIMICS) & photo].assign(label_binary=0.0)
    rng = np.random.default_rng(seed)

    def cap(d):   # at most PER_GROUP from one video / figure / page / post, chosen at random
        return d.sample(frac=1, random_state=int(rng.integers(1 << 31))).groupby("group").head(PER_GROUP)

    mf, mim = cap(mf), cap(mim)
    want = mf["source"].value_counts()
    parts = [mf]
    for src, n in want.items():
        pool = mim[mim["source"] == src]
        parts.append(pool.sample(min(POOL * n, len(pool)), random_state=int(rng.integers(1 << 31))))
    out = pd.concat(parts, ignore_index=True)
    out["rank"] = rng.permutation(len(out))
    return out


def ita(path: Path) -> float:
    """Median individual typology angle of the skin-coloured pixels (YCrCb box; the whole image
    when under 5% of it passes). A rough tone for matching the classes, not a skin type."""
    im = cv2.imread(str(path))
    if im is None:
        return float("nan")
    h, w = im.shape[:2]
    s = 256 / max(h, w)
    im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    mask = (ycc[..., 1] >= 133) & (ycc[..., 1] <= 180) & (ycc[..., 2] >= 77) & (ycc[..., 2] <= 135)
    if mask.mean() < 0.05:
        mask = np.ones_like(mask)
    L, b = lab[..., 0][mask] * 100 / 255, lab[..., 2][mask] - 128
    return float(np.median(np.degrees(np.arctan2(L - 50, b))))


def tone(a: float) -> str:
    if np.isnan(a):
        return "unknown"
    return "dark" if a <= TONE_CUTS[0] else "medium" if a <= TONE_CUTS[1] else "light"


def read_review(path: Path = REVIEW) -> dict:
    if not path.exists():
        return {}
    r = pd.read_csv(path)
    return dict(zip(r["derm1m_file"], r["verdict"]))


def match(sel: pd.DataFrame) -> pd.Series:
    """True for the photos used: every usable mycosis fungoides photo, and rashes in `rank` order,
    as many per (source, tone) as there are mycosis fungoides photos. A shortfall is made up from
    the same tone in other sources, so tone is matched first."""
    ok = sel["exclude_reason"].isna()
    used = ok & (sel["label_binary"] == 1)
    need = sel[used].groupby(["source", "tone"]).size()
    pool = sel[ok & (sel["label_binary"] == 0)].sort_values("rank")
    short = {}
    for (src, tn), n in need.items():
        got = pool[(pool["source"] == src) & (pool["tone"] == tn)].index[:n]
        used[got] = True
        if len(got) < n:
            short[tn] = short.get(tn, 0) + n - len(got)
    for tn, n in short.items():
        got = pool[(pool["tone"] == tn) & ~used[pool.index].to_numpy()].index[:n]
        used[got] = True
        print(f"tone {tn}: {n} rashes short in their own source, {len(got)} taken from other sources", flush=True)
    return used


def _safe_name(member: str) -> str:
    stem, ext = Path(member).stem, Path(member).suffix.lower()
    short = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)[-60:]
    return f"{short}_{hashlib.sha1(member.encode('utf-8')).hexdigest()[:10]}{ext if ext in ('.jpg', '.jpeg', '.png') else '.img'}"


def extract(sel: pd.DataFrame, root: Path) -> pd.Series:
    """Copies each chosen image out of its zip under a name of our own (never the archive's
    path), so nothing in an archive decides where a file is written. -> path under data_root."""
    out = pd.Series(index=sel.index, dtype=object)
    for folder, idx in sel.groupby(sel["filename"].str.split("/").str[0]).groups.items():
        dst = root / RAW / "images" / folder
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(root / RAW / ZIP_OF[folder]) as z:
            names = set(z.namelist())
            for i in idx:
                member = sel.at[i, "filename"].split("/", 1)[1]
                if member not in names:
                    continue
                rel = (RAW / "images" / folder / _safe_name(member)).as_posix()
                if not (root / rel).exists():
                    (root / rel).write_bytes(z.read(member))
                out[i] = rel
    return out


def image_check(path: Path) -> str | None:
    try:
        with Image.open(path) as im:
            w, h = im.size
            im.convert("RGB").load()
    except Exception:
        return "unreadable"
    if min(w, h) < MIN_SIDE:
        return "small_image"
    if max(w, h) / min(w, h) > MAX_ASPECT:
        return "panel_strip"
    return None


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


DEDUP_CACHE = EXTRA_DIR / "derm1m_dedup.json"


def near_duplicates(paths: list[str], groups: list[str], other: list[str], other_why: list[str], cfg: dict, root: Path):
    """-> (why: None or other_why of a matching outside image, per photo; roots: index of each photo's
    union-find root, joining source groups and near-duplicates inside Derm1M). The search over
    every validation, test and training image takes half an hour, so its result is kept in
    DEDUP_CACHE and reused while the photos, the images compared and the settings are the same."""
    sig = hashlib.sha1(json.dumps([paths, groups, other, other_why, cfg["dedup"]["phash_max_distance"],
                                   cfg["dedup"]["embed_min_cosine"]]).encode("utf-8")).hexdigest()
    if DEDUP_CACHE.exists():
        c = json.loads(DEDUP_CACHE.read_text(encoding="utf-8"))
        if c["sig"] == sig:
            print("near-duplicates: reused", DEDUP_CACHE, flush=True)
            return np.array(c["why"], dtype=object), c["roots"]
    n = len(paths)
    why, uf = np.full(n, None, dtype=object), _UF(n)
    for g in pd.Series(groups).groupby(groups).indices.values():
        for k in g[1:]:
            uf.union(int(g[0]), int(k))
    full = [root / p for p in paths] + [root / p for p in other]
    pairs = list(hash_pairs(phash_ints(full), cfg["dedup"]["phash_max_distance"]))
    pairs += list(cosine_pairs(embed(full), cfg["dedup"]["embed_min_cosine"]))
    for i, j in pairs:
        i, j = sorted((int(i), int(j)))
        if i < n <= j:
            if why[i] != "dup_of_eval":   # a match with an evaluation image outranks one with a training image
                why[i] = other_why[j - n]
        elif j < n:
            uf.union(i, j)
    roots = [uf.find(i) for i in range(n)]
    DEDUP_CACHE.write_text(json.dumps({"sig": sig, "why": why.tolist(), "roots": roots}), encoding="utf-8")
    return why, roots


def build(cfg: dict, seed: int = 0) -> pd.DataFrame:
    root = Path(cfg["data_root"])
    meta = pd.read_csv(root / RAW / "Derm1M_v2_pretrain.csv", low_memory=False)
    sel = select(meta, seed)
    print(f"chosen: {int(sel['label_binary'].sum())} mycosis fungoides, {int((sel['label_binary'] == 0).sum())} rashes", flush=True)
    sel["path"] = extract(sel, root)
    sel["exclude_reason"] = np.where(sel["path"].isna(), "missing_image", None).astype(object)
    ok = sel["exclude_reason"].isna()
    sel.loc[ok, "exclude_reason"] = [image_check(root / p) for p in sel.loc[ok, "path"]]
    print(sel["exclude_reason"].value_counts(dropna=False).to_string(), flush=True)

    # near-duplicates: against every validation / test image (manifest and outside sets), against
    # every training image (a derm1m_test photo already trained on would flatter the test, and one
    # in train_extra adds nothing), and inside Derm1M
    keep = sel[sel["exclude_reason"].isna()].reset_index()
    allm = pd.concat([pd.read_csv(cfg["manifest"], low_memory=False),
                      pd.read_csv("data/external.csv", low_memory=False)], ignore_index=True)
    ev = allm[~allm["split"].isin(NOT_EVALUATION | {EXTRA_SPLIT})]
    tr = [allm[allm["split"] == "train"]]
    for name in TRAINED_EXTRAS:
        d = pd.read_csv(EXTRA_DIR / f"{name}.csv", low_memory=False)
        tr.append(d[d["split"] == EXTRA_SPLIT])
    other = pd.concat([ev.assign(_why="dup_of_eval"), *[t.assign(_why="dup_of_train") for t in tr]], ignore_index=True)
    n = len(keep)
    why, roots = near_duplicates(list(keep["path"]), list(keep["group"]), list(other["path"]), list(other["_why"]), cfg, root)
    print(pd.Series(why).value_counts().to_string(), f"(of {n}; {len(ev)} validation / test and "
          f"{len(other) - len(ev)} training images compared): dropped", flush=True)
    dup = pd.notna(why)
    sel.loc[keep.loc[dup, "index"], "exclude_reason"] = why[dup]
    keep["group_id"] = ["derm1m_" + keep.at[roots[i], "group"] for i in range(n)]
    sel["group_id"] = "derm1m_" + sel["group"]
    sel.loc[keep["index"], "group_id"] = keep["group_id"].to_numpy()

    # skin tone, the review's verdicts, then the rashes matched to the mycosis fungoides photos
    sel["ita"] = np.nan
    sel.loc[keep["index"], "ita"] = [ita(root / p) for p in keep["path"]]
    sel["tone"] = sel["ita"].map(tone)
    verdict = sel["filename"].map(read_review()).astype(object)
    sel["review"] = verdict
    drop = verdict.isin(DROP_VERDICTS) & sel["exclude_reason"].isna()
    sel.loc[drop, "exclude_reason"] = "review_" + verdict[drop]
    used = match(sel)
    sel["split"] = np.where(used, EXTRA_SPLIT, np.where(sel["exclude_reason"].isna(), SPARE_SPLIT, "excluded"))
    print(pd.crosstab([sel["split"], sel["label_binary"]], sel["tone"]).to_string(), flush=True)

    # split by group, 30% of each label's groups held back
    rng = np.random.default_rng(seed)
    for lab in (0.0, 1.0):
        groups = sel.loc[(sel["split"] == EXTRA_SPLIT) & (sel["label_binary"] == lab), "group_id"].unique()
        test = rng.choice(groups, int(round(0.3 * len(groups))), replace=False)
        sel.loc[sel["group_id"].isin(test) & (sel["split"] == EXTRA_SPLIT), "split"] = TEST_SPLIT

    out = pd.DataFrame({
        "image_id": "derm1m_" + sel["filename"].map(lambda f: hashlib.sha1(f.encode("utf-8")).hexdigest()[:16]),
        "group_id": sel["group_id"], "dataset": "derm1m", "path": sel["path"],
        "diagnosis": sel["label"],
        "label_4class": np.where(sel["label_binary"] == 1, "exclude_rare_cancer", "exclude"),
        "label_fine": np.where(sel["label_binary"] == 1, "rare_cancer", "unlabelled"),
        "label_binary": sel["label_binary"],
        "label_cancer": sel["label_binary"].map({1.0: "cancer", 0.0: "not_cancer"}),
        "fst": np.nan, "fst_bucket": np.nan, "site": sel["body_location"], "is_acral": False,
        "patient_id": sel["group_id"], "exclude_reason": sel["exclude_reason"], "split": sel["split"],
        "source": sel["source"], "derm1m_file": sel["filename"], "caption": sel["caption"],
        "ita": sel["ita"], "tone": sel["tone"], "review": sel["review"],
    })
    return out


def contact_sheet(df: pd.DataFrame, root: Path, label: float, path: Path, n: int = 48, seed: int = 0) -> None:
    d = df[(df["label_binary"] == label) & df["split"].isin([EXTRA_SPLIT, TEST_SPLIT])]
    d = d.sample(min(n, len(d)), random_state=seed).sort_values("source")
    cols, cell = 8, 180
    rows = -(-len(d) // cols)
    from PIL import ImageDraw
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 16)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (_, r) in enumerate(d.iterrows()):
        im = Image.open(root / r["path"]).convert("RGB")
        im.thumbnail((cell - 4, cell - 4))
        x, y = (k % cols) * cell, (k // cols) * (cell + 16)
        sheet.paste(im, (x + 2, y + 2))
        draw.text((x + 3, y + cell), f"{r['source'][:14]} {r['diagnosis'][:12]}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)


def review_sheets(df: pd.DataFrame, root: Path, out_dir: Path = Path("reports/derm1m_review"),
                  cols: int = 10, rows: int = 7, cell: int = 160) -> int:
    """Numbered sheets of the used photos without a verdict, and index.csv mapping each number to
    its Derm1M file. Both classes are mixed in one random order, unlabelled, so a verdict cannot
    lean on the label."""
    from PIL import ImageDraw, ImageFont
    d = df[df["split"].isin([EXTRA_SPLIT, TEST_SPLIT]) & df["review"].isna()].sample(frac=1, random_state=0)
    d = d.reset_index(drop=True)
    if out_dir.exists():
        for f in out_dir.glob("*"):
            f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    d.assign(n=d.index)[["n", "derm1m_file", "path"]].to_csv(out_dir / "index.csv", index=False)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    per = cols * rows
    for s in range(0, len(d), per):
        part = d.iloc[s:s + per]
        sheet = Image.new("RGB", (cols * cell, -(-len(part) // cols) * cell), "white")
        draw = ImageDraw.Draw(sheet)
        for k, (i, r) in enumerate(part.iterrows()):
            im = Image.open(root / r["path"]).convert("RGB")
            im.thumbnail((cell - 4, cell - 4))
            x, y = (k % cols) * cell, (k // cols) * cell
            sheet.paste(im, (x + 2, y + 2))
            draw.rectangle([x + 2, y + 2, x + 44, y + 22], fill="yellow")
            draw.text((x + 4, y + 2), str(i), fill="black", font=font)
        sheet.save(out_dir / f"sheet_{s // per:02d}.jpg", quality=85)
    return len(d)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", action="store_true", help="also write reports/derm1m_{mf,rashes}.jpg")
    ap.add_argument("--review-sheets", action="store_true", help="also write reports/derm1m_review/ (photos without a verdict)")
    a = ap.parse_args(argv)
    cfg = load_data_config("configs/data.yaml")
    df = build(cfg)
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(pd.crosstab([df["split"], df["label_cancer"]], df["source"], margins=True).to_string())
    if a.sheet:
        root = Path(cfg["data_root"])
        contact_sheet(df, root, 1.0, Path("reports/derm1m_mf.jpg"))
        contact_sheet(df, root, 0.0, Path("reports/derm1m_rashes.jpg"))
    if a.review_sheets:
        print(f"{review_sheets(df, Path(cfg['data_root']))} photos to review in reports/derm1m_review/")


if __name__ == "__main__":
    main()
