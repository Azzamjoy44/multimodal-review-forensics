"""
build_yelp_multimodal_features.py
---------------------------------
Foundation for the MULTI-MODAL Yelp fake-review detector (thesis Option A).

The text-only Yelp models plateau at ~66-69% macro F1 because Yelp's labels are
BEHAVIORAL (its spam filter), and that signal lives in reviewer/business METADATA,
not the words. This script recovers that metadata for every review in the existing
`yelp_split.csv` (which only has text/label/split) by joining back to the raw
YelpZip / YelpNYC / YelpChi files, then engineers behavioral features.

Crucially:
  * Aggregate features (reviewer history, business stats) are computed over the
    FULL raw data (all ~1M reviews), not just the balanced split — so counts /
    averages / burst rates are real, not subsampled artifacts.
  * Each output row keeps the SAME split assignment (train/val/test) as
    yelp_split.csv, so the multi-modal model is directly comparable to the
    text-only baselines on the identical test set (no leakage, same grouping).

Output: data/yelp_multimodal_features.csv
    columns: text, label, split, source, <behavioral features...>

RUN
    python build_yelp_multimodal_features.py
"""

import os
import re
import csv
import math
from collections import defaultdict
from datetime import datetime, date

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
SPLIT_CSV = os.path.join(DATA_DIR, "yelp_split.csv")
OUT_CSV   = os.path.join(DATA_DIR, "yelp_multimodal_features.csv")

YELPZIP_META = os.path.join(DATA_DIR, "YelpZip-20260521T184341Z-3-001", "YelpZip", "metadata")
YELPZIP_REV  = os.path.join(DATA_DIR, "YelpZip-20260521T184341Z-3-001", "YelpZip", "reviewContent")
YELPNYC_META = os.path.join(DATA_DIR, "YelpNYC-20260521T184345Z-3-001", "YelpNYC", "metadata")
YELPNYC_REV  = os.path.join(DATA_DIR, "YelpNYC-20260521T184345Z-3-001", "YelpNYC", "reviewContent")
CHI_RES_META = os.path.join(DATA_DIR, "YelpChi-20260521T184346Z-3-001", "YelpChi", "output_meta_yelpResData_NRYRcleaned.txt")
CHI_RES_REV  = os.path.join(DATA_DIR, "YelpChi-20260521T184346Z-3-001", "YelpChi", "output_review_yelpResData_NRYRcleaned.txt")
CHI_HOT_META = os.path.join(DATA_DIR, "YelpChi-20260521T184346Z-3-001", "YelpChi", "output_meta_yelpHotelData_NRYRcleaned.txt")
CHI_HOT_REV  = os.path.join(DATA_DIR, "YelpChi-20260521T184346Z-3-001", "YelpChi", "output_review_yelpHotelData_NRYRcleaned.txt")


def log(m):
    print(m, flush=True)


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def _day_ordinal(date_str, fmt):
    try:
        d = datetime.strptime(date_str, fmt).date()
        return d.toordinal()
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# 1) Read every raw review into a flat list of records (full data)
#    record = dict(user, biz, rating, day, source, norm, text)
# --------------------------------------------------------------------------- #
def read_zip_style(meta_path, rev_path, prefix, records):
    if not os.path.exists(meta_path):
        log(f"  {prefix}: not found, skipping"); return 0
    n = 0
    with open(meta_path, encoding="utf-8", errors="ignore") as mf, \
         open(rev_path, encoding="utf-8", errors="ignore") as rf:
        for ml, rl in zip(mf, rf):
            m = ml.strip().split("\t")
            r = rl.strip().split("\t")
            if len(m) < 5 or len(r) < 4:
                continue
            text = "\t".join(r[3:]).strip()
            if not text:
                continue
            try:
                rating = float(m[2])
            except ValueError:
                continue
            records.append({
                "user":   f"{prefix}_{m[0].strip()}",
                "biz":    f"{prefix}_{m[1].strip()}",
                "rating": rating,
                "day":    _day_ordinal(m[4].strip(), "%Y-%m-%d"),
                "source": prefix,
                "norm":   normalize(text),
                "text":   text,
            })
            n += 1
    log(f"  {prefix}: {n:,} raw reviews")
    return n


def read_chi_style(meta_path, rev_path, prefix, records):
    if not os.path.exists(meta_path):
        log(f"  {prefix}: not found, skipping"); return 0
    n = 0
    with open(meta_path, encoding="utf-8", errors="ignore") as mf, \
         open(rev_path, encoding="utf-8", errors="ignore") as rf:
        for ml, rl in zip(mf, rf):
            parts = ml.strip().split()
            text  = rl.strip()
            if len(parts) < 9 or not text:
                continue
            try:
                rating = float(parts[8])
            except ValueError:
                continue
            records.append({
                # cols: date[0] review_id[1] reviewer_id[2] product_id[3] label[4] ... rating[8]
                "user":   f"{prefix}_{parts[2]}",   # FIXED: parts[2]=reviewer_id (was parts[1]=review_id → every chi reviewer looked like a singleton)
                "biz":    f"{prefix}_{parts[3]}",
                "rating": rating,
                "day":    _day_ordinal(parts[0], "%m/%d/%Y"),
                "source": prefix,
                "norm":   normalize(text),
                "text":   text,
            })
            n += 1
    log(f"  {prefix}: {n:,} raw reviews")
    return n


def _std(vals):
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))


def main():
    # -- split lookup: norm_text -> (label, split) ------------------------- #
    split_lookup = {}
    with open(SPLIT_CSV, encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.DictReader(f):
            split_lookup[normalize(row["text"])] = (int(row["label"]), row["split"])
    log(f"yelp_split.csv: {len(split_lookup):,} unique normalized texts")

    # -- read all raw reviews ---------------------------------------------- #
    log("reading raw Yelp files (full data) ...")
    records = []
    read_zip_style(YELPZIP_META, YELPZIP_REV, "zip",     records)
    read_zip_style(YELPNYC_META, YELPNYC_REV, "nyc",     records)
    read_chi_style(CHI_RES_META, CHI_RES_REV, "chi_res", records)
    read_chi_style(CHI_HOT_META, CHI_HOT_REV, "chi_hot", records)
    log(f"total raw reviews: {len(records):,}")

    # -- aggregates over FULL data ----------------------------------------- #
    log("building reviewer / business aggregates over full data ...")
    user_ratings = defaultdict(list)   # user -> [rating]
    user_days    = defaultdict(list)   # user -> [day ordinal]
    user_bizs    = defaultdict(list)   # user -> [biz]
    biz_ratings  = defaultdict(list)   # biz  -> [rating]
    for r in records:
        user_ratings[r["user"]].append(r["rating"])
        if r["day"] is not None:
            user_days[r["user"]].append(r["day"])
        user_bizs[r["user"]].append(r["biz"])
        biz_ratings[r["biz"]].append(r["rating"])

    # precompute per-user / per-biz scalars
    user_count   = {u: len(v) for u, v in user_ratings.items()}
    user_avg     = {u: sum(v) / len(v) for u, v in user_ratings.items()}
    user_std     = {u: _std(v) for u, v in user_ratings.items()}
    user_fpos    = {u: sum(1 for x in v if x >= 4) / len(v) for u, v in user_ratings.items()}
    user_fext    = {u: sum(1 for x in v if x in (1.0, 5.0)) / len(v) for u, v in user_ratings.items()}
    def max_per_day(days):
        if not days:
            return 0
        c = defaultdict(int)
        for d in days:
            c[d] += 1
        return max(c.values())
    user_burst   = {u: max_per_day(d) for u, d in user_days.items()}
    user_span    = {u: (max(d) - min(d) + 1) if d else 1 for u, d in user_days.items()}
    biz_count    = {b: len(v) for b, v in biz_ratings.items()}
    biz_avg      = {b: sum(v) / len(v) for b, v in biz_ratings.items()}
    biz_std      = {b: _std(v) for b, v in biz_ratings.items()}
    # per (user,biz) count -> reviews by same user on same business
    ub_count = defaultdict(int)
    for r in records:
        ub_count[(r["user"], r["biz"])] += 1

    # -- emit feature rows for reviews present in the split ---------------- #
    feat_cols = [
        "rating", "is_extreme", "review_word_len", "review_char_len",
        "rating_dev_from_biz", "abs_rating_dev_from_biz",
        "user_review_count", "user_avg_rating", "user_rating_std",
        "user_frac_positive", "user_frac_extreme", "user_is_singleton",
        "user_max_reviews_per_day", "user_reviews_per_day", "user_reviews_on_this_biz",
        "biz_review_count", "biz_avg_rating", "biz_rating_std",
    ]
    header = ["text", "label", "split", "source"] + feat_cols

    seen = set()
    matched = 0
    by_split = defaultdict(int)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(header)
        for r in records:
            norm = r["norm"]
            hit = split_lookup.get(norm)
            if hit is None or norm in seen:
                continue
            seen.add(norm)
            label, split = hit
            u, b, rating = r["user"], r["biz"], r["rating"]
            span = user_span.get(u, 1)
            feats = [
                rating,
                1 if rating in (1.0, 5.0) else 0,
                len(r["text"].split()),
                len(r["text"]),
                round(rating - biz_avg[b], 4),
                round(abs(rating - biz_avg[b]), 4),
                user_count[u],
                round(user_avg[u], 4),
                round(user_std[u], 4),
                round(user_fpos[u], 4),
                round(user_fext[u], 4),
                1 if user_count[u] == 1 else 0,
                user_burst[u],
                round(user_count[u] / max(span, 1), 5),
                ub_count[(u, b)],
                biz_count[b],
                round(biz_avg[b], 4),
                round(biz_std[b], 4),
            ]
            w.writerow([r["text"], label, split, r["source"]] + feats)
            matched += 1
            by_split[split] += 1

    log(f"\nmatched {matched:,} / {len(split_lookup):,} split rows "
        f"({matched / len(split_lookup):.1%} coverage)")
    log(f"by split: {dict(by_split)}")
    log(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
