# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
build_human_fake_dataset.py — assemble a HUMAN-WRITTEN deceptive-review corpus from the
two gold-standard sources, for training/evaluating the text fake-detector models on
*human* deception (distinct from the AI/CG fakes Model A handles, and from Yelp's
behavioural labels).

Sources (label is encoded in the file path):
  * Ott Deceptive Opinion Spam Corpus v1.4 (hotels; deceptive=MTurk, truthful=Web/TripAdvisor)
  * Li et al. 2014 cross-domain (hotel / restaurant / doctor; deceptive=MTurk+expert, truthful)

Rule: any path containing "decept" -> fake (1); "truthful" -> genuine (0). __MACOSX junk
and ._* files are skipped. Li's hotel reuses Ott's hotel reviews, so we DEDUP on normalized
text across the combined set (exact-dup removal). Stratified 80/10/10 split by label.

Output: data/human_fake/human_fake_reviews.csv  (text, label, source, domain, polarity, writer, split)
RUN:    python build_human_fake_dataset.py
"""

import os
import re
import csv
import glob
from collections import Counter

from sklearn.model_selection import train_test_split

HERE = os.path.dirname(__file__)
BASE = os.path.join(HERE, "data", "human_fake")
OTT  = os.path.join(BASE, "ott", "op_spam_v1.4")
LI   = os.path.join(BASE, "li2014", "deception_dataset")
OUT  = os.path.join(BASE, "human_fake_reviews.csv")


def _norm(t):
    return re.sub(r"\s+", " ", t.lower()).strip()


def _meta(path):
    p = path.replace("\\", "/").lower()
    # NB: match "deceptive" (the label folders are deceptive_*), NOT "decept" — the dataset
    # ROOT folder "deception_dataset" contains "decept" and would mislabel every Li review.
    label  = 1 if "deceptive" in p else (0 if "truthful" in p else None)
    domain = ("hotel" if "hotel" in p else "restaurant" if "restaurant" in p
              else "doctor" if "doctor" in p else "hotel")           # Ott is all hotel
    polarity = "negative" if "negative" in p else "positive" if "positive" in p else ""
    writer = ("expert" if "expert" in p else "mturk" if ("mturk" in p or "turker" in p)
              else "web" if label == 0 else "")
    return label, domain, polarity, writer


def collect(root, source, rows, seen):
    n = 0
    for f in glob.glob(os.path.join(root, "**", "*.txt"), recursive=True):
        if "__MACOSX" in f or os.path.basename(f).startswith("._"):
            continue
        label, domain, polarity, writer = _meta(f)
        if label is None:
            continue
        try:
            text = open(f, encoding="utf-8", errors="replace").read().strip()
        except Exception:
            continue
        if "\x00" in text:           # skip binary junk (stray .DS_Store-type files)
            continue
        if len(text) < 20:
            continue
        key = _norm(text)
        if key in seen:                       # dedup (Li hotel reuses Ott hotel)
            continue
        seen.add(key)
        rows.append({"text": text, "label": label, "source": source,
                     "domain": domain, "polarity": polarity, "writer": writer})
        n += 1
    print(f"{source}: kept {n:,} reviews")


def main():
    rows, seen = [], set()
    collect(OTT, "ott",  rows, seen)
    collect(LI,  "li2014", rows, seen)
    print(f"\ntotal after dedup: {len(rows):,}")
    print("  by label:", dict(Counter(r["label"] for r in rows)))
    print("  by source:", dict(Counter(r["source"] for r in rows)))
    print("  by domain:", dict(Counter(r["domain"] for r in rows)))

    y = [r["label"] for r in rows]
    idx = list(range(len(rows)))
    tr, tmp = train_test_split(idx, test_size=0.20, stratify=y, random_state=42)
    va, te  = train_test_split(tmp, test_size=0.50, stratify=[y[i] for i in tmp], random_state=42)
    split = {}
    for i in tr: split[i] = "train"
    for i in va: split[i] = "val"
    for i in te: split[i] = "test"

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "source", "domain", "polarity", "writer", "split"])
        for i, r in enumerate(rows):
            w.writerow([r["text"], r["label"], r["source"], r["domain"], r["polarity"], r["writer"], split[i]])
    print(f"\nwrote {OUT}")
    print(f"  split: train {len(tr):,} / val {len(va):,} / test {len(te):,}")


if __name__ == "__main__":
    main()
