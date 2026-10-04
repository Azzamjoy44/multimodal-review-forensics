import os
import csv

csv.field_size_limit(10 ** 7)

# Multimodal CSV = the frontend reviews + the 17 behavioral features Model B needs
# (built by prepare_yelp_frontend_multimodal.py). Falls back to the plain frontend
# CSV if the multimodal one is absent (Model B then can't score, but the app loads).
_MULTIMODAL_CSV = os.path.join(os.path.dirname(__file__), "data", "yelp_frontend_multimodal.csv")
_PLAIN_CSV      = os.path.join(os.path.dirname(__file__), "data", "yelp_frontend_reviews.csv")
_FRONTEND_CSV   = _MULTIMODAL_CSV if os.path.exists(_MULTIMODAL_CSV) else _PLAIN_CSV

# behavioral feature columns carried into each review entry (for Model B). Stored as
# floats; absent/blank → 0.0. user_reviews_on_this_biz is present in the CSV but Model
# B drops it, so it's harmless if carried.
BEHAV_COLS = [
    "is_extreme", "review_word_len", "review_char_len",
    "rating_dev_from_biz", "abs_rating_dev_from_biz",
    "user_review_count", "user_avg_rating", "user_rating_std",
    "user_frac_positive", "user_frac_extreme", "user_is_singleton",
    "user_max_reviews_per_day", "user_reviews_per_day", "user_reviews_on_this_biz",
    "biz_review_count", "biz_avg_rating", "biz_rating_std",
]

_business_index: dict = {}  # prefixed business_id → list[dict]
_user_index:     dict = {}  # user_id → list[dict]
_loaded = False


def _ensure_loaded():
    global _loaded
    if _loaded:
        return
    if not os.path.exists(_FRONTEND_CSV):
        print(f"Warning: {_FRONTEND_CSV} not found. "
              f"Run prepare_yelp_frontend.py to generate it.")
        _loaded = True
        return

    with open(_FRONTEND_CSV, encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.DictReader(f):
            bid = row["business_id"]
            uid = row.get("user_id", "")
            lbl = row.get("ground_truth_label", "")
            entry = {
                "review_id":          row["review_id"],
                "review_text":        row["review_text"],
                "rating":             float(row["rating"]) if row.get("rating") else None,
                "user_id":            uid,
                "time":               int(row["time"]) if row.get("time") else 0,
                "business_id":        bid,
                "ground_truth_label": int(lbl) if lbl in ("0", "1") else None,
                "source":             row.get("source", ""),
            }
            # carry behavioral features (Model B) when present
            for c in BEHAV_COLS:
                v = row.get(c, "")
                try:
                    entry[c] = float(v) if v not in ("", None) else 0.0
                except (ValueError, TypeError):
                    entry[c] = 0.0
            _business_index.setdefault(bid, []).append(entry)
            if uid:
                _user_index.setdefault(uid, []).append(entry)

    for bid in _business_index:
        _business_index[bid].sort(key=lambda r: r["time"])
    for uid in _user_index:
        _user_index[uid].sort(key=lambda r: r["time"])

    total     = sum(len(v) for v in _business_index.values())
    n_fake    = sum(1 for v in _business_index.values()
                    for r in v if r["ground_truth_label"] == 1)
    n_genuine = total - n_fake
    _loaded = True
    print(f"Yelp frontend: {total:,} reviews "
          f"({n_fake:,} fake, {n_genuine:,} genuine) "
          f"across {len(_business_index):,} businesses, "
          f"{len(_user_index):,} users.")


def get_reviews_by_business_id(business_id: str, limit: int = 20, offset: int = 0) -> list:
    _ensure_loaded()
    # stable order -> [offset:offset+limit] lets the frontend fetch only NEW reviews
    # when the requested count grows (incremental scoring cache).
    return list(_business_index.get(str(business_id), []))[offset:offset + limit]


def get_reviews_by_user_id(user_id: str) -> list:
    _ensure_loaded()
    return list(_user_index.get(str(user_id), []))


def get_popular_business_ids(n: int = 10) -> list:
    _ensure_loaded()
    return sorted(_business_index, key=lambda b: len(_business_index[b]), reverse=True)[:n]


def get_test_stats() -> dict:
    _ensure_loaded()
    total  = sum(len(v) for v in _business_index.values())
    n_fake = sum(1 for v in _business_index.values()
                 for r in v if r["ground_truth_label"] == 1)
    return {
        "total":      total,
        "fake":       n_fake,
        "genuine":    total - n_fake,
        "businesses": len(_business_index),
        "users":      len(_user_index),
    }
