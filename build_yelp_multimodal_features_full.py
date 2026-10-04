"""
build_yelp_multimodal_features_full.py — multi-modal features for the FULL YelpZip+NYC+Chi
dataset (~1.03M reviews, TRUE ~12-13% fake distribution), for retraining Model B on all data
instead of the balanced 178k yelp_split subset.

Differences vs build_yelp_multimodal_features.py:
  * labels come from the RAW spam flags for EVERY review (zip/nyc col 3 == '-1' -> fake;
    chi col 4 == 'Y' -> fake), not just the yelp_split subset.
  * split: yelp_split test/val rows stay test/val (so the TEST set is IDENTICAL to all the
    other Yelp models -> directly comparable); everything else -> train.
  * FIXES the YelpChi reviewer column (parts[2] = reviewer_id, not parts[1] = review_id) so
    chi reviewer aggregates aren't degenerate.
  * emits ALL ~1.03M reviews (deduped on normalized text).

Aggregates (reviewer/business) are over the full pool, same definitions as the 178k builder.
Output: data/yelp_multimodal_features_full.csv   RUN: python build_yelp_multimodal_features_full.py
"""

import os
import csv
from collections import defaultdict

import build_yelp_multimodal_features as M   # paths + normalize/_day_ordinal/_std helpers

OUT = os.path.join(M.DATA_DIR, "yelp_multimodal_features_full.csv")


def read_zip(meta, rev, prefix, recs):
    if not os.path.exists(meta):
        print(f"  {prefix}: missing"); return
    n = 0
    with open(meta, encoding="utf-8", errors="ignore") as mf, open(rev, encoding="utf-8", errors="ignore") as rf:
        for ml, rl in zip(mf, rf):
            m = ml.strip().split("\t"); r = rl.strip().split("\t")
            if len(m) < 5 or len(r) < 4:
                continue
            text = "\t".join(r[3:]).strip()
            if not text:
                continue
            try: rating = float(m[2])
            except ValueError: continue
            recs.append({"user": f"{prefix}_{m[0].strip()}", "biz": f"{prefix}_{m[1].strip()}",
                         "rating": rating, "day": M._day_ordinal(m[4].strip(), "%Y-%m-%d"),
                         "source": prefix, "norm": M.normalize(text), "text": text,
                         "label": 1 if m[3].strip() == "-1" else 0})   # -1 = fake (filtered)
            n += 1
    print(f"  {prefix}: {n:,}")


def read_chi(meta, rev, prefix, recs):
    if not os.path.exists(meta):
        print(f"  {prefix}: missing"); return
    n = 0
    with open(meta, encoding="utf-8", errors="ignore") as mf, open(rev, encoding="utf-8", errors="ignore") as rf:
        for ml, rl in zip(mf, rf):
            p = ml.strip().split(); text = rl.strip()
            if len(p) < 9 or not text:
                continue
            try: rating = float(p[8])
            except ValueError: continue
            # cols: date[0] review_id[1] reviewer_id[2] product_id[3] label[4] ... rating[8]
            recs.append({"user": f"{prefix}_{p[2]}", "biz": f"{prefix}_{p[3]}",   # FIXED: p[2] reviewer
                         "rating": rating, "day": M._day_ordinal(p[0], "%m/%d/%Y"),
                         "source": prefix, "norm": M.normalize(text), "text": text,
                         "label": 1 if p[4].strip().upper() == "Y" else 0})
            n += 1
    print(f"  {prefix}: {n:,}")


def main():
    # yelp_split: norm -> split (test/val/train); only used to pin test/val for comparability
    split_of = {}
    with open(M.SPLIT_CSV, encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.DictReader(f):
            split_of[M.normalize(row["text"])] = row["split"]
    print(f"yelp_split: {len(split_of):,} norms")

    print("reading raw Yelp (full) ...")
    recs = []
    read_zip(M.YELPZIP_META, M.YELPZIP_REV, "zip", recs)
    read_zip(M.YELPNYC_META, M.YELPNYC_REV, "nyc", recs)
    read_chi(M.CHI_RES_META, M.CHI_RES_REV, "chi_res", recs)
    read_chi(M.CHI_HOT_META, M.CHI_HOT_REV, "chi_hot", recs)
    print(f"total raw: {len(recs):,}")

    print("aggregates over full pool ...")
    ur, ud, ub, br = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    ubc = defaultdict(int)
    for r in recs:
        ur[r["user"]].append(r["rating"])
        if r["day"] is not None: ud[r["user"]].append(r["day"])
        ub[r["user"]].append(r["biz"]); br[r["biz"]].append(r["rating"])
        ubc[(r["user"], r["biz"])] += 1
    ucount = {u: len(v) for u, v in ur.items()}
    uavg = {u: sum(v)/len(v) for u, v in ur.items()}
    ustd = {u: M._std(v) for u, v in ur.items()}
    ufpos = {u: sum(1 for x in v if x >= 4)/len(v) for u, v in ur.items()}
    ufext = {u: sum(1 for x in v if x in (1.0, 5.0))/len(v) for u, v in ur.items()}
    def mpd(days):
        c = defaultdict(int)
        for d in days: c[d] += 1
        return max(c.values()) if c else 0
    uburst = {u: mpd(d) for u, d in ud.items()}
    uspan = {u: (max(d)-min(d)+1) if d else 1 for u, d in ud.items()}
    bcount = {b: len(v) for b, v in br.items()}
    bavg = {b: sum(v)/len(v) for b, v in br.items()}
    bstd = {b: M._std(v) for b, v in br.items()}

    feat_cols = ["rating", "is_extreme", "review_word_len", "review_char_len",
                 "rating_dev_from_biz", "abs_rating_dev_from_biz",
                 "user_review_count", "user_avg_rating", "user_rating_std",
                 "user_frac_positive", "user_frac_extreme", "user_is_singleton",
                 "user_max_reviews_per_day", "user_reviews_per_day", "user_reviews_on_this_biz",
                 "biz_review_count", "biz_avg_rating", "biz_rating_std"]
    seen = set(); by_split = defaultdict(int); n_fake = 0
    with open(OUT, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out); w.writerow(["text", "label", "split", "source"] + feat_cols)
        for r in recs:
            if r["norm"] in seen or len(r["norm"]) < 1:
                continue
            seen.add(r["norm"])
            u, b, rating = r["user"], r["biz"], r["rating"]
            split = split_of.get(r["norm"]) if split_of.get(r["norm"]) in ("test", "val") else "train"
            span = uspan.get(u, 1)
            feats = [rating, 1 if rating in (1.0, 5.0) else 0, len(r["text"].split()), len(r["text"]),
                     round(rating-bavg[b], 4), round(abs(rating-bavg[b]), 4),
                     ucount[u], round(uavg[u], 4), round(ustd[u], 4), round(ufpos[u], 4),
                     round(ufext[u], 4), 1 if ucount[u] == 1 else 0, uburst.get(u, 0),
                     round(ucount[u]/max(span, 1), 5), ubc[(u, b)],
                     bcount[b], round(bavg[b], 4), round(bstd[b], 4)]
            w.writerow([r["text"], r["label"], split, r["source"]] + feats)
            by_split[split] += 1; n_fake += r["label"]
    tot = sum(by_split.values())
    print(f"\nwrote {OUT}: {tot:,} reviews | fake {n_fake:,} ({n_fake/max(tot,1):.1%}) | split {dict(by_split)}")


if __name__ == "__main__":
    main()
