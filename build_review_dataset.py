"""
build_review_dataset.py — assemble the review / non-review training corpus.

POSITIVE class (label 1, reviews): 50k each from 6 domains
    amazon  data/Reviews.csv         column 'Text'
    imdb    data/IMDB Dataset.csv     column 'review'
    yelp    data/yelp_split.csv       column 'text'
    steam   HF SirSkandrani/steam_reviews_clean    column 'text'   (streamed)
    books   HF McAuley-Lab/Amazon-Reviews-2023 Books.jsonl 'text' (HTTP stream)
    apps    HF sealuzh/app_reviews                  column 'review' (streamed)

NEGATIVE class (label 0, non-reviews): data/non_reviews.csv (built by
    download_negatives.py — ~380k rows across 18 sources: 6 topic-matched
    descriptions + questions/news/encyclopedic + 8 broad genres)

The two classes are then downsampled to an exact 50/50 balance before the split.

Output: data/review_detection.csv with columns  text, label, source, split
    (split in {train, val, test}, stratified 80/10/10 by label).
"""

import os
import csv
import json
import random
import urllib.request

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_url
from sklearn.model_selection import train_test_split

from download_negatives import clean, DATA_DIR

OUT_CSV = os.path.join(DATA_DIR, "review_detection.csv")
NEG_CSV = os.path.join(DATA_DIR, "non_reviews.csv")
PER_DOMAIN = 50_000
SEED = 42

random.seed(SEED)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# positive collectors  (return list of cleaned strings, up to `target`)
# ---------------------------------------------------------------------------
def from_csv(path, col, target, seen):
    df = pd.read_csv(path, usecols=[col])
    df = df.sample(frac=1, random_state=SEED)        # shuffle for diversity
    out = []
    for v in df[col]:
        t = clean(v)
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= target:
            break
    return out


def from_hf_stream(repo, field, target, seen):
    ds = load_dataset(repo, split="train", streaming=True)
    out = []
    for row in ds:
        t = clean(row.get(field))
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= target:
            break
    return out


def from_amazon_jsonl(filename, field, target, seen):
    """Stream an Amazon-Reviews-2023 raw jsonl file over HTTP (datasets 4.x
    can't load its script-based configs)."""
    url = hf_hub_url("McAuley-Lab/Amazon-Reviews-2023", filename, repo_type="dataset")
    req = urllib.request.Request(url, headers={"User-Agent": "thesis-downloader"})
    out = []
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            try:
                row = json.loads(raw)
            except Exception:
                continue
            t = clean(row.get(field))
            if not t:
                continue
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
            if len(out) >= target:
                break
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    rows = []            # (text, label, source)
    seen = set()

    # --- negatives first, so positives can be deduped against them ----------
    neg = pd.read_csv(NEG_CSV)
    for t, src in zip(neg["text"], neg["source"]):
        k = str(t).lower()
        if k in seen:
            continue
        seen.add(k)
        rows.append((t, 0, src))
    log(f"negatives: {len(rows)}")

    # --- positives ----------------------------------------------------------
    pos_specs = [
        ("amazon", lambda: from_csv(os.path.join(DATA_DIR, "Reviews.csv"), "Text", PER_DOMAIN, seen)),
        ("imdb",   lambda: from_csv(os.path.join(DATA_DIR, "IMDB Dataset.csv"), "review", PER_DOMAIN, seen)),
        ("yelp",   lambda: from_csv(os.path.join(DATA_DIR, "yelp_split.csv"), "text", PER_DOMAIN, seen)),
        ("steam",  lambda: from_hf_stream("SirSkandrani/steam_reviews_clean", "text", PER_DOMAIN, seen)),
        ("books",  lambda: from_amazon_jsonl("raw/review_categories/Books.jsonl", "text", PER_DOMAIN, seen)),
        ("apps",   lambda: from_hf_stream("sealuzh/app_reviews", "review", PER_DOMAIN, seen)),
    ]
    n_pos = 0
    for name, fn in pos_specs:
        got = fn()
        for t in got:
            rows.append((t, 1, name))
        n_pos += len(got)
        log(f"positive[{name}]: {len(got)}")
    log(f"positives total: {n_pos} | grand total: {len(rows)}")

    # --- build dataframe ----------------------------------------------------
    df = pd.DataFrame(rows, columns=["text", "label", "source"])

    # --- balance to exactly 50/50 (downsample the majority class) -----------
    n_min = int(df["label"].value_counts().min())
    df = (pd.concat([df[df.label == 0].sample(n=n_min, random_state=SEED),
                     df[df.label == 1].sample(n=n_min, random_state=SEED)])
            .reset_index(drop=True))
    log(f"\nbalanced to 50/50: {n_min} per class ({2*n_min} total)")

    # --- length report (the cue we must not let dominate) -------------------
    wl = df["text"].str.split().str.len()
    log("\nword-length by class (min/median/mean/95p/max):")
    for lab, name in [(1, "review    "), (0, "non-review")]:
        s = wl[df["label"] == lab]
        log(f"  {name}: {s.min():4d} / {int(s.median()):4d} / {s.mean():6.1f} / "
            f"{int(s.quantile(.95)):4d} / {s.max():5d}")

    log("\nper-source counts:")
    log(df.groupby(["label", "source"]).size().to_string())

    # --- stratified 80/10/10 split by label --------------------------------
    idx = df.index.to_numpy()
    tr, tmp = train_test_split(idx, test_size=0.20, random_state=SEED,
                               stratify=df["label"])
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED,
                              stratify=df.loc[tmp, "label"])
    df["split"] = "train"
    df.loc[va, "split"] = "val"
    df.loc[te, "split"] = "test"
    log("\nsplit sizes:")
    log(df.groupby(["split", "label"]).size().to_string())

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    log(f"\nwrote {OUT_CSV}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
