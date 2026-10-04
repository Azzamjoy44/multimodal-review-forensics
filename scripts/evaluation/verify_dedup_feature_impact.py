# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
verify_dedup_feature_impact.py — does deduping-before-aggregating actually change the 17
behavioral features on the SERVED (test+val) rows? Computes each row's features TWO ways
(aggregates over the raw ~1.03M pool vs over the text-deduped 678k pool) and diffs them.
No model retrain — pure feature comparison. RUN: python verify_dedup_feature_impact.py
"""
import csv
from collections import defaultdict
import numpy as np
import build_yelp_multimodal_features as M
import build_yelp_multimodal_features_full as MF
import model_b

FEAT = model_b.FEAT_COLS  # the 17 served features (excludes user_reviews_on_this_biz)


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
    A = {}
    A["ucount"] = {u: len(v) for u, v in ur.items()}
    A["uavg"]   = {u: sum(v)/len(v) for u, v in ur.items()}
    A["ustd"]   = {u: M._std(v) for u, v in ur.items()}
    A["ufpos"]  = {u: sum(1 for x in v if x >= 4)/len(v) for u, v in ur.items()}
    A["ufext"]  = {u: sum(1 for x in v if x in (1.0, 5.0))/len(v) for u, v in ur.items()}
    A["uburst"] = {u: mpd(d) for u, d in ud.items()}
    A["uspan"]  = {u: (max(d)-min(d)+1) if d else 1 for u, d in ud.items()}
    A["bcount"] = {b: len(v) for b, v in br.items()}
    A["bavg"]   = {b: sum(v)/len(v) for b, v in br.items()}
    A["bstd"]   = {b: M._std(v) for b, v in br.items()}
    return A


def feats(r, A):
    u, b, rating = r["user"], r["biz"], r["rating"]
    span = A["uspan"].get(u, 1)
    return {
        "rating": rating, "is_extreme": 1 if rating in (1.0, 5.0) else 0,
        "review_word_len": len(r["text"].split()), "review_char_len": len(r["text"]),
        "rating_dev_from_biz": round(rating-A["bavg"][b], 4),
        "abs_rating_dev_from_biz": round(abs(rating-A["bavg"][b]), 4),
        "user_review_count": A["ucount"][u], "user_avg_rating": round(A["uavg"][u], 4),
        "user_rating_std": round(A["ustd"][u], 4), "user_frac_positive": round(A["ufpos"][u], 4),
        "user_frac_extreme": round(A["ufext"][u], 4), "user_is_singleton": 1 if A["ucount"][u] == 1 else 0,
        "user_max_reviews_per_day": A["uburst"].get(u, 0),
        "user_reviews_per_day": round(A["ucount"][u]/max(span, 1), 5),
        "biz_review_count": A["bcount"][b], "biz_avg_rating": round(A["bavg"][b], 4),
        "biz_rating_std": round(A["bstd"][b], 4),
    }


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
    print(f"  raw recs: {len(recs):,}", flush=True)

    A_raw = aggregates(recs)                                   # current behaviour (over raw 1.03M)
    seen, deduped = set(), []
    for r in recs:
        if r["norm"] in seen: continue
        seen.add(r["norm"]); deduped.append(r)
    A_ded = aggregates(deduped)                                # dedup-before-aggregate (over 678k)
    print(f"  deduped recs: {len(deduped):,}", flush=True)

    # served rows = the deduped rows that land in yelp_split test/val (what the frontend serves)
    served = [r for r in deduped if split_of.get(r["norm"]) in ("test", "val")]
    print(f"  served (test+val) rows: {len(served):,}\n", flush=True)

    changed = 0
    by_source = defaultdict(int)
    maxdelta = defaultdict(float)
    for r in served:
        fr, fd = feats(r, A_raw), feats(r, A_ded)
        row_changed = False
        for c in FEAT:
            d = abs(float(fr[c]) - float(fd[c]))
            if d > 1e-9:
                row_changed = True; maxdelta[c] = max(maxdelta[c], d)
        if row_changed:
            changed += 1; by_source[r["user"].split("_")[0]] += 1

    print("=" * 60)
    print(f"served rows whose 17 features CHANGE if deduped-first: {changed:,} / {len(served):,} "
          f"({changed/len(served):.2%})")
    print(f"  by source dataset: {dict(by_source)}")
    print("  per-feature max |delta| (only features that moved):")
    for c in FEAT:
        if maxdelta[c] > 0:
            print(f"    {c:<26} {maxdelta[c]:.4g}")
    if not maxdelta:
        print("    (none — features identical under both aggregation pools)")
    print("=" * 60)


if __name__ == "__main__":
    main()
