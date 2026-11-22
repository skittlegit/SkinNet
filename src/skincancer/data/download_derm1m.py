"""Download Derm1M (Yan et al., ICCV 2025; CC BY-NC 4.0) into data/raw/derm1m.

    python -m skincancer.data.download_derm1m            # metadata only (about 0.4 GB)
    python -m skincancer.data.download_derm1m --images   # metadata, then every image zip (about 34.5 GB)

The dataset is gated on Hugging Face: the account logged in on this machine must first accept its
terms at https://huggingface.co/datasets/redlessone/Derm1M. Zips are downloaded, not extracted:
extraction is a separate, checked step, since the archives come from outside the project.
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

REPO = "redlessone/Derm1M"
META = ["README.md", "Derm1M_v2_validation.csv", "Derm1M_v2_pretrain.csv", "concept.csv", "ontology.json"]
ZIPS = ["validation_data.zip", "reddit.zip", "twitter.zip", "pubmed.zip", "IIYI.zip", "note.zip",
        "public.zip", "edu.zip", "youtube.zip"]          # smallest first


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", action="store_true", help="also download the image zips")
    ap.add_argument("--out", default="data/raw/derm1m")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = META + (ZIPS if a.images else [])
    for f in files:
        try:
            p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=out)
        except GatedRepoError:
            print("No access: accept the terms at https://huggingface.co/datasets/" + REPO + " with this machine's account.")
            return 1
        print("ok", f, Path(p).stat().st_size, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
