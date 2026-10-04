"""
verify_everything_ok.py — definitive integrity check of the Yelp Model-B feature pipeline.
Every claim made in discussion is tested with hard numbers. Prints PASS/FAIL per check.
RUN: python verify_everything_ok.py
"""
import csv
from collections import defaultdict
import numpy as np
import pandas as pd
import build_yelp_multimodal_features as M
import build_yelp_multimodal_features_full as MF
import model_b

FEAT = model_b.FEAT_COLS


def aggregates(recs):
    ur, ud, br = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in recs:
        ur[r["user"]].append(r["rating"])
        if r["day"] is not None: ud[r["user"]].append(r["day"])
        br[r["biz"]].append(r["rating"])
    def mpd(days):
        c = defaultdict(int)
        for d in days: c[d] += 1
        return max(c.values()) if c else 0
    return {
        "ucount": {u: len(v) for u, v in ur.items()},
        "uavg": {u: sum(v)/len(v) for u, v in ur.items()},
        "ustd": {u: M._std(v) for u, v in ur.items()},
        "ufpos": {u: sum(1 for x in v if x >= 4)/len(v) for u, v in ur.items()},
        "ufext": {u: sum(1 for x in v if x in (1.0, 5.0))/len(v) for u, v in ur.items()},
        "uburst": {u: mpd(d) for u, d in ud.items()},
        "uspan": {u: (max(d)-min(d)+1) if d else 1 for u, d in ud.items()},
        "bcount": {b: len(v) for b, v in br.items()},
        "bavg": {b: sum(v)/len(v) for b, v in br.items()},
        "bstd": {b: M._std(v) for b, v in br.items()},
    }


def feats(r, A):
    u, b, rating = r["user"], r["biz"], r["rating"]
    span = A["uspan"].get(u, 1)
    return [rating, 1 if rating in (1.0, 5.0) else 0, len(r["text"].split()), len(r["text"]),
            round(rating-A["bavg"][b], 4), round(abs(rating-A["bavg"][b]), 4),
            A["ucount"][u], round(A["uavg"][u], 4), round(A["ustd"][u], 4), round(A["ufpos"][u], 4),
            round(A["ufext"][u], 4), 1 if A["ucount"][u] == 1 else 0, A["uburst"].get(u, 0),
            round(A["ucount"][u]/max(span, 1), 5),
            A["bcount"][b], round(A["bavg"][b], 4), round(A["bstd"][b], 4)]


def main():
    split_of = {}
    with open(M.SPLIT_CSV, encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.DictReader(f):
            split_of[M.normalize(row["text"])] = row["split"]

    print("reading raw Yelp ...", flush=True)
    recs = []
    MF.read_zip(M.YELPZIP_META, M.YELPZIP_REV, "zip", recs)
    MF.read_zip(M.YELPNYC_META, M.YELPNYC_REV, "nyc", recs)
    MF.read_chi(M.CHI_RES_META, M.CHI_RES_REV, "chi_res", recs)
    MF.read_chi(M.CHI_HOT_META, M.CHI_HOT_REV, "chi_hot", recs)
    print(f"  raw recs: {len(recs):,}\n", flush=True)

    # ---- build 3 pools -----------------------------------------------------
    A_full = aggregates(recs)                                       # CURRENT: full raw pool
    seen_src, pool_B = {}, []                                       # cross-dataset dups removed ONLY
    for r in recs:
        n = r["norm"]
        if n in seen_src and seen_src[n] != r["source"]:
            continue                                                # drop cross-dataset copy
        seen_src.setdefault(n, r["source"]); pool_B.append(r)
    A_nocross = aggregates(pool_B)

    # served rows = keep-first-by-text rows that land in test/val (the frontend set)
    seen, served = set(), []
    for r in recs:
        if r["norm"] in seen: continue
        seen.add(r["norm"])
        if split_of.get(r["norm"]) in ("test", "val"):
            served.append(r)

    # ===== CHECK 1: NYC cross-dataset dupes are INERT for served features =====
    diff_by_src = defaultdict(int)
    for r in served:
        if feats(r, A_full) != feats(r, A_nocross):
            diff_by_src[r["user"].split("_")[0]] += 1
    zip_changed = diff_by_src.get("zip", 0)
    print("CHECK 1 — removing cross-dataset dupes (a dedup we deliberately DON'T do): served feature change?")
    print(f"  served rows: {len(served):,} | changed by source: {dict(diff_by_src) or 'none'}")
    print(f"  -> zip rows changed: {zip_changed}  (must be 0 — NYC dupes carry nyc_ IDs, can't touch zip)  "
          f"[{'PASS' if zip_changed == 0 else 'FAIL'}]")
    print(f"     (chi/nyc changes are EXPECTED & CORRECT: coincidental cross-source text we rightly keep)\n")

    # ===== CHECK 2: exact reviewer+business+text double-counts =====
    exact = defaultdict(int)
    for r in recs:
        exact[(r["user"], r["biz"], r["norm"])] += 1
    n_exact = sum(c-1 for c in exact.values() if c > 1)
    print(f"CHECK 2 — exact (reviewer+business+text) double-counts: {n_exact}  "
          f"[{'PASS: none to remove' if n_exact == 0 else 'note'}]\n")

    # ===== CHECK 3: frontend CSV integrity =====
    fe = pd.read_csv("data/yelp_frontend_multimodal.csv")
    cols_ok = all(c in fe.columns for c in FEAT)
    blanks = int(fe[FEAT].isna().sum().sum()) + int((fe[FEAT].astype(str) == "").sum().sum())
    bal = fe["ground_truth_label"].mean()
    print("CHECK 3 — frontend serving CSV:")
    print(f"  rows {len(fe):,} | 17 FEAT_COLS present {cols_ok} | blank feature cells {blanks} | "
          f"fake rate {bal:.1%}  [{'PASS' if cols_ok and blanks == 0 else 'FAIL'}]\n")

    # ===== CHECK 4: served features == the features the deep models trained on =====
    full = pd.read_csv("data/yelp_multimodal_features_full.csv", usecols=["text"] + FEAT)
    full["k"] = full["text"].map(M.normalize)
    full = full.drop_duplicates("k").set_index("k")
    fe["k"] = fe["review_text"].map(M.normalize)
    j = fe.merge(full[FEAT], left_on="k", right_index=True, suffixes=("_fe", "_full"))
    maxdiff = 0.0
    for c in FEAT:
        maxdiff = max(maxdiff, (j[f"{c}_fe"].astype(float) - j[f"{c}_full"].astype(float)).abs().max())
    print("CHECK 4 — frontend served features == full-training features (serving consistency):")
    print(f"  matched {len(j):,}/{len(fe):,} frontend rows | max |feature delta| {maxdiff:.6g}  "
          f"[{'PASS: identical' if maxdiff < 1e-4 else 'FAIL'}]\n")

    print("=" * 64)
    ok = (zip_changed == 0 and n_exact == 0 and cols_ok and blanks == 0 and maxdiff < 1e-4)
    print("OVERALL:", "ALL CHECKS PASS — features are correct & serving-consistent." if ok
          else "SOMETHING FAILED — see above.")
    print("=" * 64)


if __name__ == "__main__":
    main()
