"""
load_amazon_reviews.py
----------------------
Loads reviews from data/Reviews.csv (Amazon product reviews, Kaggle).
Builds a ProductId and UserId index on first call; subsequent calls are instant.
"""

import csv
import os
import re

AMAZON_REVIEWS_FILE = os.path.join(os.path.dirname(__file__), "data", "Reviews.csv")


def _strip_html(text):
    """Remove HTML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_product_index  = None
_user_index     = None
_dataset_max_time = 0


def _ensure_loaded():
    global _product_index, _user_index, _dataset_max_time
    if _product_index is not None:
        return

    if not os.path.exists(AMAZON_REVIEWS_FILE):
        print(f"Warning: '{AMAZON_REVIEWS_FILE}' not found.")
        _product_index = {}
        _user_index    = {}
        return

    print("Loading Amazon reviews index (one-time startup) ...")
    product_idx = {}
    user_idx    = {}
    max_time    = 0

    with open(AMAZON_REVIEWS_FILE, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("ProductId", "").strip()
            uid = row.get("UserId",    "").strip()
            if not pid or not uid:
                continue
            try:
                rating = int(float(row.get("Score", "3") or "3"))
            except (ValueError, TypeError):
                rating = 3
            try:
                time = int(row.get("Time", "0") or "0")
            except (ValueError, TypeError):
                time = 0

            if time > max_time:
                max_time = time

            entry = {
                "review_id":   row.get("Id", "").strip(),
                "review_text": _strip_html(row.get("Text", "")),
                "rating":      rating,
                "product_id":  pid,
                "summary":     row.get("Summary", "").strip(),
                "user_id":     uid,
                "time":        time,
            }
            product_idx.setdefault(pid, []).append(entry)
            user_idx.setdefault(uid, []).append(entry)

    _product_index   = product_idx
    _user_index      = user_idx
    _dataset_max_time = max_time
    total = sum(len(v) for v in product_idx.values())
    print(f"Indexed {total:,} reviews across {len(product_idx):,} products and {len(user_idx):,} users.")


def get_reviews_by_product_id(product_id: str, limit: int = 100):
    """Return up to `limit` reviews for the given Amazon ProductId."""
    _ensure_loaded()
    return _product_index.get(product_id.strip(), [])[:limit]


def get_reviews_by_user_id(user_id: str):
    """Return all reviews posted by the given UserId, sorted by time."""
    _ensure_loaded()
    reviews = _user_index.get(user_id.strip(), [])
    return sorted(reviews, key=lambda r: r["time"])


def get_dataset_max_time():
    """Return the most recent Unix timestamp in the dataset."""
    _ensure_loaded()
    return _dataset_max_time


def list_product_ids(limit: int = 50):
    """Return up to `limit` ProductIds present in the dataset."""
    _ensure_loaded()
    return list(_product_index.keys())[:limit]


if __name__ == "__main__":
    _ensure_loaded()
    sample_ids = list_product_ids(5)
    print(f"\nSample product IDs: {sample_ids}\n")
    for pid in sample_ids[:2]:
        all_reviews = get_reviews_by_product_id(pid, limit=10_000)
        preview     = all_reviews[:3]
        print(f"Product {pid}  ({len(all_reviews)} reviews total)")
        for r in preview:
            print(f"  [{r['review_id']}] rating={r['rating']}  \"{r['review_text'][:80]}...\"")
        print()
