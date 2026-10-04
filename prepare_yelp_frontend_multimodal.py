"""
prepare_yelp_frontend_multimodal.py — build the Model-B-ready frontend CSV.

Takes the existing frontend review pool (data/yelp_frontend_reviews.csv, served by
load_yelp_reviews.py) and attaches the 17 engineered BEHAVIORAL features that the
multi-modal Model B needs, by joining to data/yelp_multimodal_features.csv.

Why a join (not recompute): the behavioral features (user_review_count,
biz_review_count, rating deviations, per-day burst counts, …) are aggregated over the
FULL 1.03M-review raw Yelp pool. They cannot be recomputed from the 17,891-review
frontend subset — within-subset counts would be wildly out-of-distribution (the exact
failure that killed the Tier-3 upload attempt). So we pull the already-computed values.

The frontend pool == the `test` split of yelp_multimodal_features.csv, and every row
matches uniquely by normalized review text (verified 100%, 0 missing / 0 ambiguous).

Output: data/yelp_frontend_multimodal.csv — all original frontend columns PLUS the 17
behavioral feature columns. Same row set, so any business/user stays browsable; this
is purely additive and sets up serving Model B in the Yelp section.
"""

import os
import csv

csv.field_size_limit(10 ** 7)

_DATA      = os.path.join(os.path.dirname(__file__), "data")
FRONTEND   = os.path.join(_DATA, "yelp_frontend_reviews.csv")
FEATURES   = os.path.join(_DATA, "yelp_multimodal_features.csv")
OUT        = os.path.join(_DATA, "yelp_frontend_multimodal.csv")

# the 17 behavioral feature columns (everything in the features file that isn't a
# key/meta column or already present in the frontend CSV). `rating` is already in the
# frontend file; Model B uses it too but we keep the frontend's copy.
BEHAV_COLS = [
    "is_extreme", "review_word_len", "review_char_len",
    "rating_dev_from_biz", "abs_rating_dev_from_biz",
    "user_review_count", "user_avg_rating", "user_rating_std",
    "user_frac_positive", "user_frac_extreme", "user_is_singleton",
    "user_max_reviews_per_day", "user_reviews_per_day", "user_reviews_on_this_biz",
    "biz_review_count", "biz_avg_rating", "biz_rating_std",
]


def _norm(t):
    return " ".join((t or "").split()).lower()


def main():
    # 1) index behavioral features by normalized text
    feats_by_text = {}
    dupes = 0
    with open(FEATURES, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in BEHAV_COLS if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"features file missing columns: {missing}")
        for row in reader:
            key = _norm(row["text"])
            if key in feats_by_text:
                dupes += 1            # keep first; frontend join is unique anyway
                continue
            feats_by_text[key] = {c: row[c] for c in BEHAV_COLS}
    print(f"Indexed {len(feats_by_text):,} unique feature rows (skipped {dupes:,} dup texts).")

    # 2) read frontend reviews, attach features, write out
    with open(FRONTEND, encoding="utf-8", errors="ignore", newline="") as f:
        rows = list(csv.DictReader(f))
        in_cols = list(rows[0].keys())
    out_cols = in_cols + BEHAV_COLS

    matched = missed = 0
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for r in rows:
            feat = feats_by_text.get(_norm(r["review_text"]))
            if feat:
                r.update(feat); matched += 1
            else:
                for c in BEHAV_COLS:    # leave blank if (shouldn't happen) unmatched
                    r[c] = ""
                missed += 1
            w.writerow(r)

    print(f"Wrote {OUT}")
    print(f"  rows: {len(rows):,} | matched: {matched:,} | unmatched: {missed:,}")
    print(f"  columns: {len(out_cols)} ({len(in_cols)} original + {len(BEHAV_COLS)} behavioral)")
    if missed:
        print("  WARNING: some rows had no behavioral features (blank).")


if __name__ == "__main__":
    main()
