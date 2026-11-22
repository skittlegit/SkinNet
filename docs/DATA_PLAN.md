# Data plan: what to add after the baseline

Status (15 Nov 2026): the plan below is the original one of 12 Nov and is kept as a record. What happened to each source is in [IDEAS.md](IDEAS.md): MIDAS (checkpoint 3) and MILK10k (checkpoint 4) helped outside; Derm7pt, SCIN and ISIC 2019 did not. The search of 15 Nov for more sources is in the last section.

## Why more data

The baseline trains on 12,562 images, of which 2,220 are cancer. Two weaknesses are known before any result:

1. **Few hard cases.** Most not-cancer images are rashes and similar conditions that look little like a tumour. The hard part of the task is telling a cancer from a benign tumour that resembles it, and training holds only 733 such benign lookalikes.
2. **A shortcut.** 70% of the PAD-UFES-20 images are cancer against 13% of the Fitzpatrick17k images, and the two are different kinds of photo (smartphone against atlas). The network can score well by recognising the kind of photo.

## Candidates

Counts marked "about" are from memory and must be confirmed at download.

| Dataset | What it adds | On disk? | Cost or risk | Confidence it helps |
| --- | --- | --- | --- | --- |
| Derm7pt, clinical photos | 248 melanomas, 42 BCCs, 664 benign nevi | Yes | None: it is not a test set. No SCC. | High |
| MRA-MIDAS, 70% by patient | About 560 cancers and 400 benign tumours, all biopsy-confirmed | Yes | Shrinks our largest outside tumour test to 30%. Cannot be undone once a model has trained on it. | Fairly high |
| SCIN (Google): **tried as idea I, 13 Nov 2026, discarded** (helped at home, hurt on MIDAS) | About 10,000 phone photos from the public, almost all not cancer | Yes (2,902 photos) | Adds no cancers. Labels come from dermatologists reading photos, not biopsies. | High, for weakness 2 |
| ISIC 2019 | About 8,500 cancers and 12,900 nevi | No | Dermoscopy images (a magnifying scope pressed on the skin), which look nothing like ordinary photos. May not help on clinical photos. | Unsure |

Not planned: Dermofit (paid licence, about 400 cancers) and MED-NODE (170 images).

## Proposed order

Each step is one training run, compared with the run before it on the same test sets. A step is kept only if it improves the outside scores (DDI, the held-back part of MIDAS, DermaCon-IN), not just the score on our own `test` split.

1. **Baseline.** Fitzpatrick17k + PAD-UFES-20 only. Running now.
2. **Add Derm7pt and 70% of MIDAS.** Both are on disk. Training cancers go from 2,220 to about 3,070, and most of the additions are hard cases.
3. **Add SCIN.** Check whether false alarms on phone photos of harmless conditions fall (DermaCon-IN everyday conditions is the test for this).
4. **Add ISIC 2019,** as a with-and-without experiment.

## Open decisions

- **MIDAS:** move 70% into training, or keep all of it as a test set? DDI stays an untouched, biopsy-confirmed outside test either way.
- **Dermoscopy:** mix ISIC images into the same model, or keep the project on clinical photos only?
- **Every added dataset needs the same checks as the current ones:** diagnoses mapped through the label map, a near-duplicate search against the test sets, and splits by patient. Adding data means rebuilding `train` / `val` / `test`, so earlier runs must be repeated on the new splits to stay comparable.

## Not covered here

Skin tone. Almost all of these datasets are lighter skin, so none of them fixes the shortage of dark-skin cancers. Whether darker skin needs its own model or one combined model is the question for stage two ([DECISIONS.md](../DECISIONS.md), section 2).

## Search of 15 Nov 2026: more biopsy-confirmed clinical photos

Why: of every idea tried, only clinical photos from hospitals not yet in training (MIDAS, then MILK10k) raised an outside score. Dermoscopy (ISIC 2019, idea K) and a new mix of photos already used (idea R) did not.

The ISIC archive was queried directly (15 Nov): it holds 9,316 clinical close-ups, of which 5,719 are MILK10k and 2,298 PAD-UFES-20, both in training already; 1,241 from Memorial Sloan Kettering are almost all benign by one clinician's judgement (6 melanomas); the rest are a handful. So the archive has nothing large left of this kind.

| Source | What it is | Licence and access | Verdict |
| --- | --- | --- | --- |
| Asan and Hallym (Han et al. 2018; figshare 5406136) | Two Korean hospitals; 16,836 clinical photos in 12 diagnoses, 12,176 marked biopsied, of them 3,982 cancers | CC BY 4.0, open; but only 99 x 99 px thumbnails (full size on request to Asan's ethics committee) | **Tried as idea S and discarded** (16 Nov; DECISIONS.md, section 22): a gain at home, a loss on MIDAS |
| HIBA (Hospital Italiano de Buenos Aires; ISIC collection 175) | 349 clinical overview photos besides dermoscopy, biopsy-confirmed, Argentina | CC BY, open through ISIC | Small; worth adding together with another source |
| MCR-SL (Univ. Hospital of North Norway, 2025; Zenodo 17306338) | 877 clinical photos of 240 lesions from 60 people; excised lesions confirmed by histopathology | Open download (4.4 GB); licence not stated on the record | Small; the licence must be confirmed first |
| Dermofit (Edinburgh) | 1,300 clinical close-ups in 10 classes, every one with a confirmed diagnosis | £75 academic licence, requested from an institutional email, one research group | Good quality; needs the team or supervisor to apply |
| MED-NODE (Groningen) | 170 clinical photos: 70 melanomas, 100 nevi | Free for research | Too small alone |
| SLICE-3D (ISIC 2024) | 401,059 crops from 3-D total-body photography, 393 cancers, several centres | CC BY-NC (a CC BY subset exists) | A different kind of image with very few cancers; the same risk as idea K |
| Asan full-size images | As above, at full resolution | Request to Asan's ethics committee | The thumbnails did not help outside (idea S); whether the full-size photos would is untested. The team's choice |

## Search of 22 Nov 2026: mycosis fungoides photos

Why: mycosis fungoides, a skin lymphoma that looks like a rash, is 27 of DDI's 48 cancers on skin types V–VI. PanDerm without DDI ranked it below benign lesions (RESULTS.md, DDI in training). Our training holds 123 such photos, only 10 on types V–VI.

**On disk already.** All 182 of Fitzpatrick17k's photos are used: 123 in train, 25 in val, 34 in test. DermaCon-IN has 4, but that set is test-only. SCIN's labels are dermatologists' guesses from photos, not biopsies.

| Source | What it is | Licence and access | Verdict |
| --- | --- | --- | --- |
| Dermoscopy of Mycosis Fungoides (Mendeley r9xfjgsyrx, 2024) | 89 patients in Tunisia, skin types III–V: early mycosis fungoides confirmed by biopsy and immunohistochemistry, and psoriasis and chronic dermatitis lookalikes. Clinical and dermoscopic photos were taken of each lesion. | CC BY 4.0. A download button is shown, but which files it holds is not confirmed. | **No photos published** (checked 22 Nov): its files are only an SPSS table and the ethics approval. Asking its authors for the photos is the only way. |
| Large plaque parapsoriasis and early mycosis fungoides (Mendeley 4p88hnwyj8, 2025) | Clinical photos from Cumhuriyet University, Turkey. Count, confirmation method and skin types are not stated. | CC BY 4.0 | **Only the paper's 3 figures** (parapsoriasis, patch and plaque mycosis fungoides), downloaded to `data/raw/mf_mendeley_tr/`. Too few to train on. |
| Derm1M (ICCV 2025) | 403,563 images in 390 conditions, from YouTube, PubMed, forums, public datasets and teaching slides. | CC BY-NC 4.0, gated on Hugging Face: about 34.5 GB in 9 zips plus 0.35 GB of captions. `python -m skincancer.data.download_derm1m [--images]` once the account has accepted the terms. | Mycosis fungoides is likely among the conditions but not confirmed. Labels come from the text, not biopsies. It may hold copies of Fitzpatrick17k, DDI or other test images, so it needs the near-duplicate check against every test set. It comes from PanDerm's authors, which may overlap with PanDerm's pretraining. |
| Multimodal mycosis fungoides CNN (Br J Dermatol 2025, PubMed 40465821) | 1,157 cases, 2,452 clinical photos | Not public. Single centre: Peking Union Medical College Hospital with Beihang University (Liu Z., Zhang Y., ... Xie F., Liu J.; doi 10.1093/bjd/ljaf212). It also holds the inflammatory lookalikes. | Largest known set. The team would have to ask the corresponding authors; sharing patient photos out of China may need approvals on their side. |
