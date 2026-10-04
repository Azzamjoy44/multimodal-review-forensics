"""
build_combined_fake_dataset.py — merge the two fake-review corpora into ONE dataset for a
UNIFIED detector that flags a review fake if it is EITHER computer-generated OR human-deceptive.

  * CG/OR   (data/fake_reviews_broadened.csv): label 1 = CG (AI) fake, 0 = original/genuine
  * human   (data/human_fake/human_fake_reviews.csv): label 1 = human-deceptive, 0 = truthful

Output rows tagged fake_type in {cg_fake, human_fake, genuine} so we can report per-type
recall (does the unified model catch BOTH?). Existing train/val/test splits are PRESERVED
from each source (no re-split -> no leakage). Deduped on normalized text across the union.

Output: data/combined_fake_reviews.csv  (text, label, fake_type, domain, source, split)
RUN:    python build_combined_fake_dataset.py
"""

import os
import re
import csv
from collections import Counter, defaultdict

import pandas as pd

HERE = os.path.dirname(__file__)
CGOR = os.path.join(HERE, "data", "fake_reviews_broadened.csv")
HUM  = os.path.join(HERE, "data", "human_fake", "human_fake_reviews.csv")
OUT  = os.path.join(HERE, "data", "combined_fake_reviews.csv")


def _norm(t):
    return re.sub(r"\s+", " ", str(t).lower()).strip()


def main():
    cg = pd.read_csv(CGOR)
    hu = pd.read_csv(HUM)
    cg["fake_type"] = cg["label"].map({1: "cg_fake", 0: "genuine"})
    hu["fake_type"] = hu["label"].map({1: "human_fake", 0: "genuine"})
    cg["origin"] = "cgor"; hu["origin"] = "human"
    cols = ["text", "label", "fake_type", "domain", "source", "split", "origin"]
    df = pd.concat([cg[cols], hu[cols]], ignore_index=True)

    # dedup on normalized text (keep first); guards against any cross-corpus overlap
    df["_n"] = df["text"].map(_norm)
    before = len(df)
    df = df[df["_n"].str.len() >= 20].drop_duplicates("_n").drop(columns="_n").reset_index(drop=True)
    print(f"rows: {before:,} -> {len(df):,} after dedup/short-filter")
    print("  by label:", dict(Counter(df.label)))
    print("  by fake_type:", dict(Counter(df.fake_type)))
    print("  by split:", dict(Counter(df.split)))

    # pre-flight: domain x fake_type (a domain that is fake-only is a SHORTCUT risk)
    print("\ndomain x fake_type (watch for fake-only domains):")
    piv = defaultdict(lambda: Counter())
    for d, ft in zip(df.domain, df.fake_type):
        piv[d][ft] += 1
    print(f"  {'domain':<12}{'cg_fake':>9}{'human_fake':>12}{'genuine':>10}")
    for d in sorted(piv):
        c = piv[d]
        print(f"  {str(d):<12}{c['cg_fake']:>9}{c['human_fake']:>12}{c['genuine']:>10}")

    df.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
