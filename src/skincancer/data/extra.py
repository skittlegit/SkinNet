"""Training-only extra data, kept outside the frozen manifest.

    python -m skincancer.data.extra milk10k

Writes data/extra/<name>.csv in the manifest's columns with split = "train_extra". The
manifest, the groups and every validation / test split stay untouched, so models trained
with and without the extra data are scored on exactly the same images. A config opts in
with `data.extra_train: [milk10k]`. Extra images that near-duplicate any validation or test
image (same thresholds as data/dedup.py) are dropped.

Derm7pt (Kawahara et al., IEEE JBHI 2019; CC BY-NC-ND 4.0, never shown in figures): one
clinical (non-dermoscopic) photo per case from an Italian dermoscopy atlas. Mostly nevi and
melanomas, some BCC, no SCC. It records no skin type.

MILK10k (ISIC collection 425, MILK study team, CC BY-NC 4.0; idea Q): clinical close-ups of
biopsied lesions from five centres. Only histopathology-confirmed lesions are used.
    python -m skincancer.data.extra milk10k --download && python -m skincancer.data.extra milk10k

ISIC 2019 (ISIC collection 65: HAM10000, BCN_20000, MSK; CC licences per image; idea K):
dermoscopy, a different kind of image from every evaluation set. A fixed subsample of 2,500
cancers and 2,500 not cancer, one image per lesion, stored resized to 512 px on the short side.
    python -m skincancer.data.extra isic2019 --download && python -m skincancer.data.extra isic2019

Asan and Hallym (Han et al., J Invest Dermatol 2018; figshare 5406136, CC BY 4.0; idea S):
clinical photos from two Korean hospitals, published only as contact sheets of 99 x 99 px
thumbnails, one sheet per diagnosis; the download cuts them into single photos. Only photos
marked as biopsied are used, and a fixed subsample of 2,500 cancers and 2,500 not cancer.
    python -m skincancer.data.extra asan --download && python -m skincancer.data.extra asan
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_data_config
from ..labels import CANCER_TYPES, TUMOUR, map_diagnosis
from .dedup import cosine_pairs, embed, hash_pairs, phash_ints

EXTRA_DIR = Path("data/extra")
EXTRA_SPLIT = "train_extra"
NOT_EVALUATION = {"train", "excluded"}   # duplicates of these cannot leak into a score


def derm7pt_rows(root: Path) -> pd.DataFrame:
    base = next(p for p in (root / "derm7pt").rglob("meta.csv")).parent.parent   # release_v0/
    meta = pd.read_csv(base / "meta" / "meta.csv")
    rel = base.relative_to(root)
    label = meta["diagnosis"].map(lambda d: map_diagnosis(d)[0])
    ids = "derm7pt_" + meta["case_num"].astype(str)
    df = pd.DataFrame({
        "image_id": ids, "group_id": ids, "dataset": "derm7pt",
        "path": [(rel / "images" / p).as_posix() for p in meta["clinic"]],
        "diagnosis": meta["diagnosis"], "label_4class": label,
        "label_fine": label.where(label.isin(TUMOUR), "unlabelled"),
    })
    # vascular lesions, melanosis and "miscellaneous" have no tumour label here: left out
    df["label_binary"] = label.map(lambda l: 1.0 if l in CANCER_TYPES else 0.0 if l == "benign" else np.nan)
    df["label_cancer"] = df["label_binary"].map({1.0: "cancer", 0.0: "not_cancer"})
    df["fst"] = np.nan
    df["fst_bucket"] = np.nan
    df["site"] = meta.get("location")
    df["is_acral"] = meta.get("location", pd.Series("", index=meta.index)).astype(str).str.contains(
        "acral|palm|sole", case=False)
    df["patient_id"] = ids
    return df


MILK_API = "https://api.isic-archive.com/api/v2/images/search/?collections=425&limit=100"   # ISIC collection 425


def milk10k_download(root: Path, workers: int = 16) -> None:
    """Fetch the metadata of every MILK10k image, then the clinical close-ups (about 5,240)."""
    import json
    from concurrent.futures import ThreadPoolExecutor
    import socket
    from urllib.request import urlopen, urlretrieve
    socket.setdefaulttimeout(60)
    d = root / "milk10k"
    (d / "images").mkdir(parents=True, exist_ok=True)
    meta = d / "metadata.json"
    if not meta.exists():
        rows, url = [], MILK_API
        while url:
            with urlopen(url) as r:
                page = json.load(r)
            rows += page["results"]
            url = page.get("next")
        meta.write_text(json.dumps(rows), encoding="utf-8")
    rows = json.loads(meta.read_text(encoding="utf-8"))
    todo = [(r["files"]["full"]["url"], d / "images" / f"{r['isic_id']}.jpg") for r in rows
            if r["metadata"]["acquisition"].get("image_type") == "clinical: close-up"
            and not (d / "images" / f"{r['isic_id']}.jpg").exists()]
    print(f"{len(todo)} clinical close-ups to fetch", flush=True)

    def get(job) -> bool:
        url, dst = job
        tmp = dst.with_suffix(".part")
        try:
            urlretrieve(url, tmp)
            tmp.replace(dst)
            return True
        except Exception as e:   # reported; a second run fetches what is missing
            print("failed", dst.name, e, flush=True)
            return False

    with ThreadPoolExecutor(workers) as ex:
        done = 0
        for i, ok in enumerate(ex.map(get, todo), 1):
            done += ok
            if i % 500 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} tried, {done} fetched", flush=True)


HIBA_API = "https://api.isic-archive.com/api/v2/images/search/?collections=175&limit=100"   # ISIC collection 175
CLINICAL_TYPES = ("clinical: close-up", "clinical: overview")


def isic_collection_download(d: Path, api: str, image_types: tuple[str, ...], workers: int = 16) -> None:
    """Fetch the metadata of every image in an ISIC collection, then the images of the given types."""
    import json
    from concurrent.futures import ThreadPoolExecutor
    import socket
    from urllib.request import urlopen, urlretrieve
    socket.setdefaulttimeout(60)
    (d / "images").mkdir(parents=True, exist_ok=True)
    meta = d / "metadata.json"
    if not meta.exists():
        rows, url = [], api
        while url:
            with urlopen(url) as r:
                page = json.load(r)
            rows += page["results"]
            url = page.get("next")
        meta.write_text(json.dumps(rows), encoding="utf-8")
    rows = json.loads(meta.read_text(encoding="utf-8"))
    todo = [(r["files"]["full"]["url"], d / "images" / f"{r['isic_id']}.jpg") for r in rows
            if r["metadata"]["acquisition"].get("image_type") in image_types
            and not (d / "images" / f"{r['isic_id']}.jpg").exists()]
    print(f"{len(rows)} images in the collection, {len(todo)} of types {image_types} to fetch", flush=True)

    def get(job) -> bool:
        url, dst = job
        tmp = dst.with_suffix(".part")
        try:
            urlretrieve(url, tmp)
            tmp.replace(dst)
            return True
        except Exception as e:   # reported; a second run fetches what is missing
            print("failed", dst.name, e, flush=True)
            return False

    with ThreadPoolExecutor(workers) as ex:
        done = sum(ex.map(get, todo))
    print(f"{done} of {len(todo)} fetched", flush=True)


def hiba_download(root: Path) -> None:
    """HIBA (Hospital Italiano de Buenos Aires; ISIC collection 175, CC BY): the clinical photos only."""
    isic_collection_download(root / "hiba", HIBA_API, CLINICAL_TYPES)


def _fetch(url: str, dst: Path) -> None:
    from urllib.request import Request, urlopen
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print("have", dst, flush=True)
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"})) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dst)
    print("fetched", dst, dst.stat().st_size, flush=True)


def mednode_download(root: Path) -> None:
    """MED-NODE (University Medical Center Groningen): 70 melanomas and 100 nevi, clinical photos. Zip only."""
    _fetch("https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/complete_mednode_dataset.zip",
           root / "mednode" / "complete_mednode_dataset.zip")


def mcrsl_download(root: Path) -> None:
    """MCR-SL (University Hospital of North Norway; Zenodo 17306338; licence not stated on the record). Zip only."""
    _fetch("https://zenodo.org/records/17306338/files/MCR-SL_dataset.zip?download=1", root / "mcrsl" / "MCR-SL_dataset.zip")


def isic_binary(clin: pd.DataFrame, need_biopsy: bool) -> tuple[pd.Series, pd.Series, np.ndarray]:
    """ISIC archive labels -> (most specific diagnosis, mapped label, cancer 1 / not 0 / NaN).
    Cancer when ISIC-DX says malignant and the most specific diagnosis maps to a cancer type;
    not cancer when it says benign; actinic keratoses (never trained on) and indeterminate
    lesions are NaN, and so is anything not biopsied when need_biopsy."""
    specific = clin[[c for c in ("diagnosis_5", "diagnosis_4", "diagnosis_3", "diagnosis_2") if c in clin]].bfill(axis=1).iloc[:, 0]
    label = specific.map(lambda s: map_diagnosis(s)[0])
    top = clin["diagnosis_1"]
    ok = (clin.get("diagnosis_confirm_type", pd.Series(index=clin.index, dtype=object)) == "histopathology") \
        if need_biopsy else pd.Series(True, index=clin.index)
    binary = np.where(ok & (top == "Malignant") & label.isin(CANCER_TYPES), 1.0,
                      np.where(ok & (top == "Benign") & (label != "ak"), 0.0, np.nan))
    return specific, label, binary


ISIC2019_API = "https://api.isic-archive.com/api/v2/images/search/?collections=65&limit=100"   # Challenge 2019: Training
ISIC2019_PER_CLASS = 2500   # idea K (DECISIONS.md, section 20): 2,500 cancers + 2,500 not cancer
ISIC2019_SHORT_SIDE = 512   # stored resized: some originals are 6000 px wide


def _isic2019_meta(root: Path) -> list[dict]:
    import json
    return json.loads((root / "isic2019" / "metadata.json").read_text(encoding="utf-8"))


def isic2019_rows(root: Path) -> pd.DataFrame:
    """ISIC 2019 challenge training set (ISIC collection 65; HAM10000, BCN_20000 and MSK, each
    image under its own CC licence, recorded in `licence`): dermoscopy. Every image is labelled;
    a fixed random subsample is kept (idea K): one image per lesion, then 2,500 cancers and
    2,500 not cancer, drawn with seed 0. Labels as MILK10k, without needing a biopsy, since the
    challenge's benign ground truth includes follow-up and expert consensus."""
    rows = _isic2019_meta(root)
    clin = pd.DataFrame([r["metadata"].get("clinical", {}) for r in rows])
    specific, label, binary = isic_binary(clin, need_biopsy=False)
    iid = pd.Series([r["isic_id"] for r in rows])
    lesion = clin["lesion_id"] if "lesion_id" in clin else pd.Series(np.nan, index=clin.index)
    group = ("isic2019_" + lesion.astype(str)).where(lesion.notna(), "isic2019_" + iid)
    df = pd.DataFrame({
        "image_id": "isic2019_" + iid, "group_id": group, "dataset": "isic2019",
        "path": "isic2019/images/" + iid + ".jpg",
        "diagnosis": specific, "label_4class": label,
        "label_fine": label.where(label.isin(TUMOUR), np.where(binary == 0.0, "other_benign_tumour", "unlabelled")),
        "licence": [r.get("copyright_license") for r in rows],
    })
    df["label_binary"] = binary
    df["label_cancer"] = df["label_binary"].map({1.0: "cancer", 0.0: "not_cancer"})
    df["fst"] = np.nan
    df["fst_bucket"] = np.nan
    df["site"] = clin.get("anatom_site_1")
    df["is_acral"] = df["site"].astype(str).str.contains("acral|palm|sole", case=False)
    df["patient_id"] = clin["patient_id"].where(clin["patient_id"].notna(), group) if "patient_id" in clin else group
    # the subsample: one image per lesion (the first by id), then a fixed draw per class
    rng = np.random.default_rng(0)
    labelled = df[df["label_binary"].notna()].sort_values("image_id").drop_duplicates("group_id")
    picked = []
    for _, cls in labelled.groupby("label_binary"):
        picked += list(cls["image_id"].to_numpy()[rng.permutation(len(cls))[:ISIC2019_PER_CLASS]])
    df["in_subsample"] = df["image_id"].isin(picked)
    return df


def isic2019_download(root: Path, workers: int = 16) -> None:
    """Fetch the metadata of all 25,331 images, then only the subsample, resized."""
    import io
    import json
    from concurrent.futures import ThreadPoolExecutor
    import socket
    from urllib.request import urlopen
    from PIL import Image
    socket.setdefaulttimeout(60)
    d = root / "isic2019"
    (d / "images").mkdir(parents=True, exist_ok=True)
    meta = d / "metadata.json"
    if not meta.exists():
        rows, url = [], ISIC2019_API
        while url:
            with urlopen(url) as r:
                page = json.load(r)
            rows += page["results"]
            url = page.get("next")
        meta.write_text(json.dumps(rows), encoding="utf-8")
    urls = {"isic2019_" + r["isic_id"]: r["files"]["full"]["url"] for r in _isic2019_meta(root)}
    df = isic2019_rows(root)
    todo = [(urls[i], root / p) for i, p in df.loc[df["in_subsample"], ["image_id", "path"]].itertuples(index=False)
            if not (root / p).exists()]
    print(f"{len(todo)} images to fetch", flush=True)

    def get(job) -> bool:
        url, dst = job
        try:
            with urlopen(url) as r:
                im = Image.open(io.BytesIO(r.read())).convert("RGB")
            s = ISIC2019_SHORT_SIDE / min(im.size)
            if s < 1:
                im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            tmp = dst.with_suffix(".part")
            im.save(tmp, "JPEG", quality=95)
            tmp.replace(dst)
            return True
        except Exception as e:   # reported; a second run fetches what is missing
            print("failed", dst.name, e, flush=True)
            return False

    with ThreadPoolExecutor(workers) as ex:
        done = 0
        for i, ok in enumerate(ex.map(get, todo), 1):
            done += ok
            if i % 500 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} tried, {done} fetched", flush=True)


ASAN_URL = "https://ndownloader.figshare.com/files/9318406"   # figshare 5406136, "Asan and Hallym (thumbnails).zip"
ASAN_MD5 = "8d98bb2a7fd0d4a0d409c8625753d205"
ASAN_PER_CLASS = 2500   # idea S (DECISIONS.md, section 22): same share of the draws as MILK10k and idea K
# sheet class -> diagnosis in the label map's words; intraepithelial carcinoma is Bowen's disease (SCC in situ)
ASAN_DIAGNOSIS = {"bcc": "basal cell carcinoma", "back-bcc": "basal cell carcinoma", "hallym-bcc": "basal cell carcinoma",
                  "scc": "squamous cell carcinoma", "intraepithelial-carcinoma": "bowen disease", "melanoma": "melanoma",
                  "nevus": "nevus", "sebk": "seborrheic keratosis", "dermatofibroma": "dermatofibroma",
                  "lentigo": "lentigo", "pyogenic-granuloma": "pyogenic granuloma", "hemangioma": "hemangioma",
                  "wart": "wart", "ak": "actinic keratosis", "actinickeratosis": "actinic keratosis"}
ASAN_OTHER_BENIGN = {"hemangioma", "wart"}   # benign growths outside the label map, as MILK10k's


def _runs(ix: np.ndarray) -> list[tuple[int, int]]:
    out, start, prev = [], ix[0], ix[0]
    for i in ix[1:]:
        if i != prev + 1:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out


def asan_download(root: Path) -> None:
    """Fetch the thumbnail sheets and cut each into its 99 x 99 px photos:
    asan_hallym/tiles/<sheet>/<row>_<col>.png."""
    import hashlib
    import urllib.request
    import zipfile
    from io import BytesIO
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    d = root / "asan_hallym"
    d.mkdir(parents=True, exist_ok=True)
    zpath = d / "asan_hallym_thumbnails.zip"
    if not zpath.exists():
        urllib.request.urlretrieve(ASAN_URL, zpath)
    if hashlib.md5(zpath.read_bytes()).hexdigest() != ASAN_MD5:
        raise ValueError(f"{zpath}: checksum does not match figshare's")
    z = zipfile.ZipFile(zpath)
    for name in sorted(n for n in z.namelist() if n.endswith(".png")):
        im = np.array(Image.open(BytesIO(z.read(name))).convert("RGBA"))
        bg = (im[..., 3] < 10) | (im[..., :3].astype(int) > 245).all(-1)
        rows = [r for r in _runs(np.where(~bg.all(1))[0]) if 80 <= r[1] - r[0] <= 130]
        cols = [c for c in _runs(np.where(~bg.all(0))[0]) if 80 <= c[1] - c[0] <= 130]
        out = d / "tiles" / name.split("/")[-1][:-4].replace("#", "_").replace(" ", "-")
        out.mkdir(parents=True, exist_ok=True)
        for i, (r0, r1) in enumerate(rows):
            for j, (c0, c1) in enumerate(cols):
                if bg[r0:r1 + 1, c0:c1 + 1].mean() > 0.5:   # empty cell at the end of the last row
                    continue
                tile = im[r0 + 1:r1 - 1, c0 + 1:c1 - 1, :3]   # without the drop-shadow edge
                Image.fromarray(np.ascontiguousarray(tile)).save(out / f"{i:03d}_{j:02d}.png")
        print(f"{name.split('/')[-1]}: {len(rows)} rows x {len(cols)} columns", flush=True)


def asan_rows(root: Path) -> pd.DataFrame:
    """Asan (train 90% and test 10% sheets) and Hallym, one dataset `asan`: two Korean hospitals.
    Sheet names carry the split, whether the photo was biopsied and the diagnosis. Cancer: BCC,
    SCC, Bowen's disease, melanoma. Not cancer: nevi, seborrhoeic keratoses, dermatofibromas,
    lentigines, pyogenic granulomas, haemangiomas, warts. Actinic keratoses and every photo not
    marked biopsied are left out. No lesion or patient ids are published, so each photo is its
    own group; several photos of one lesion sit side by side on a sheet."""
    recs = []
    for sheet in sorted((root / "asan_hallym" / "tiles").iterdir()):
        parts = sheet.name.split("_")   # L, 12dx, <split>, [biopsy | non-biopsy,] <class>
        biopsied = parts[2] == "hallym" or parts[3] == "biopsy"   # Hallym's sheets are BCCs, all biopsied (Han et al.)
        for tile in sorted(sheet.glob("*.png")):
            recs.append({"sheet": sheet.name, "cls": parts[-1], "biopsied": biopsied,
                         "path": f"asan_hallym/tiles/{sheet.name}/{tile.name}"})
    t = pd.DataFrame(recs)
    diag = t["cls"].map(ASAN_DIAGNOSIS)
    if diag.isna().any():
        raise ValueError(f"unknown Asan classes: {sorted(t.loc[diag.isna(), 'cls'].unique())}")
    label = diag.map(lambda s: map_diagnosis(s)[0])
    other = t["cls"].isin(ASAN_OTHER_BENIGN)
    binary = np.where(~t["biopsied"] | (label == "ak"), np.nan,
                      np.where(label.isin(CANCER_TYPES), 1.0, np.where(label.isin(TUMOUR) | other, 0.0, np.nan)))
    iid = "asan_" + t["sheet"].str.replace("L_12dx_", "", regex=False) + "_" + t["path"].str[-10:-4]
    df = pd.DataFrame({
        "image_id": iid, "group_id": iid, "dataset": "asan", "path": t["path"],
        "diagnosis": diag, "label_4class": label,
        "label_fine": label.where(label.isin(TUMOUR), np.where(binary == 0.0, "other_benign_tumour", "unlabelled")),
        "licence": "CC BY 4.0",
    })
    df["label_binary"] = binary
    df["label_cancer"] = df["label_binary"].map({1.0: "cancer", 0.0: "not_cancer"})
    df["fst"] = np.nan
    df["fst_bucket"] = np.nan
    df["site"] = np.nan
    df["is_acral"] = False
    df["patient_id"] = df["group_id"]
    rng = np.random.default_rng(0)
    picked = []
    for _, cls in df[df["label_binary"].notna()].sort_values("image_id").groupby("label_binary"):
        picked += list(cls["image_id"].to_numpy()[rng.permutation(len(cls))[:ASAN_PER_CLASS]])
    df["in_subsample"] = df["image_id"].isin(picked)
    return df


def milk10k_rows(root: Path) -> pd.DataFrame:
    """MILK10k (ISIC collection 425; CC BY-NC 4.0, "MILK study team"): one clinical close-up
    per lesion from five centres, 95.7% of lesions biopsied or excised. Cancer when ISIC-DX
    says malignant and the most specific diagnosis maps to a cancer type; not cancer when
    ISIC-DX says benign; indeterminate lesions and actinic keratoses (never trained on, as in
    the manifest) are left out."""
    import json
    rows = json.loads((root / "milk10k" / "metadata.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if r["metadata"]["acquisition"].get("image_type") == "clinical: close-up"]
    clin = pd.DataFrame([r["metadata"]["clinical"] for r in rows])
    ids = pd.Series(["milk10k_" + r["isic_id"] for r in rows])
    specific, label, binary = isic_binary(clin, need_biopsy=True)   # 224 rest on one clinician's opinion: left out
    df = pd.DataFrame({
        "image_id": ids, "group_id": "milk10k_" + clin["lesion_id"].astype(str), "dataset": "milk10k",
        "path": ["milk10k/images/" + r["isic_id"] + ".jpg" for r in rows],
        "diagnosis": specific, "label_4class": label,
        "label_fine": label.where(label.isin(TUMOUR), np.where(binary == 0.0, "other_benign_tumour", "unlabelled")),
    })
    df["label_binary"] = binary
    df["label_cancer"] = df["label_binary"].map({1.0: "cancer", 0.0: "not_cancer"})
    df["fst"] = np.nan        # not recorded
    df["fst_bucket"] = np.nan
    df["site"] = clin.get("anatom_site_1")
    df["is_acral"] = df["site"].astype(str).str.contains("acral|palm|sole", case=False)
    df["patient_id"] = df["group_id"]   # no patient ids are published: one photo per lesion
    return df


def drop_eval_duplicates(extra: pd.DataFrame, manifest: pd.DataFrame, root: Path, cfg: dict) -> pd.DataFrame:
    ev = manifest[~manifest["split"].isin(NOT_EVALUATION)].reset_index(drop=True)
    paths = [root / p for p in extra["path"]] + [root / p for p in ev["path"]]
    n = len(extra)
    dup = np.zeros(n, bool)
    for i, j in hash_pairs(phash_ints(paths), cfg["dedup"]["phash_max_distance"]):
        if i < n <= j:
            dup[i] = True
    for i, j in cosine_pairs(embed(paths), cfg["dedup"]["embed_min_cosine"]):
        if i < n <= j:
            dup[i] = True
    print(f"{dup.sum()} of {n} extra images near-duplicate one of {len(ev)} validation / test images: dropped")
    extra = extra.copy()
    extra.loc[dup, "exclude_reason"] = "dup_of_eval"
    return extra


def build(name: str, cfg: dict, check_duplicates: bool = True) -> pd.DataFrame:
    root = Path(cfg["data_root"])
    df = {"derm7pt": derm7pt_rows, "milk10k": milk10k_rows, "isic2019": isic2019_rows,
          "asan": asan_rows}[name](root)
    df["exclude_reason"] = pd.Series(np.nan, index=df.index, dtype=object)
    df.loc[df["label_binary"].isna(), "exclude_reason"] = "no_cancer_label"
    if "in_subsample" in df:
        df.loc[df["exclude_reason"].isna() & ~df.pop("in_subsample"), "exclude_reason"] = "not_in_subsample"
    df.loc[df["exclude_reason"].isna() & ~df["path"].map(lambda p: (root / p).exists()), "exclude_reason"] = "missing_image"
    keep = df["exclude_reason"].isna()
    if check_duplicates:
        checked = drop_eval_duplicates(df[keep].reset_index(drop=True), pd.read_csv(cfg["manifest"], low_memory=False),
                                       root, cfg)
        df = pd.concat([checked, df[~keep]], ignore_index=True)
    df["split"] = np.where(df["exclude_reason"].isna(), EXTRA_SPLIT, "excluded")
    return df


def load_extra(names: list[str], label_col: str, classes: list[str], extra_dir: Path = EXTRA_DIR) -> pd.DataFrame:
    """Rows of the named extra sets that may be trained on."""
    df = pd.concat([pd.read_csv(Path(extra_dir) / f"{n}.csv") for n in names], ignore_index=True)
    return df[(df["split"] == EXTRA_SPLIT) & df[label_col].isin(classes)].reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=["derm7pt", "milk10k", "isic2019", "asan", "hiba", "mednode", "mcrsl"])
    ap.add_argument("--download", action="store_true", help="milk10k, isic2019, asan, hiba, mednode, mcrsl: fetch the tables and photos, then stop")
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args(argv)
    cfg = load_data_config(args.data_config)
    if args.download:
        return {"milk10k": milk10k_download, "isic2019": isic2019_download,
                "asan": asan_download, "hiba": hiba_download, "mednode": mednode_download,
                "mcrsl": mcrsl_download}[args.name](Path(cfg["data_root"]))
    df = build(args.name, cfg)
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXTRA_DIR / f"{args.name}.csv", index=False)
    print(pd.crosstab(df["split"], df["label_fine"], margins=True).to_string())
    print(df["exclude_reason"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
