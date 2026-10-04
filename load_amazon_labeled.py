"""
load_amazon_labeled.py — serves LABELED fake-review browse sets for the Fake Review Detection
section, so the UI can compare predictions against ground truth (like the Yelp section) and toggle
between IN-DISTRIBUTION and OUT-OF-DISTRIBUTION data:
  * 'indist' → data/amazon_fake_labeled.csv  (held-out products test split the detector trained on)
  * 'ood'    → data/ood_hc3_labeled.csv       (HC3 ChatGPT-vs-human; NEVER trained on — OOD probe)
Each entry carries a ground_truth_label (1 = AI/fake, 0 = human/genuine).
"""
import os
import csv

csv.field_size_limit(10 ** 7)

_DATA = os.path.join(os.path.dirname(__file__), "data")
_FILES = {
    "indist": "amazon_fake_labeled.csv",
    "ood":    "ood_hc3_labeled.csv",
}
_cache: dict = {}     # dataset -> list[review dict]


def _load(dataset: str) -> list:
    dataset = dataset if dataset in _FILES else "indist"
    if dataset in _cache:
        return _cache[dataset]
    rows = []
    path = os.path.join(_DATA, _FILES[dataset])
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="ignore", newline="") as f:
            for row in csv.DictReader(f):
                lbl = row.get("ground_truth_label", "")
                rows.append({
                    "review_id":          row["review_id"],
                    "review_text":        row["review_text"],
                    "rating":             None, "user_id": "", "time": 0,
                    "ground_truth_label": int(lbl) if lbl in ("0", "1") else None,
                    "source":             row.get("source", ""),
                })
        n_fake = sum(1 for r in rows if r["ground_truth_label"] == 1)
        print(f"Labeled '{dataset}': {len(rows):,} reviews ({n_fake:,} fake / {len(rows) - n_fake:,} genuine).")
    else:
        print(f"Warning: {path} not found. Run the matching build_*_labeled.py to generate it.")
    _cache[dataset] = rows
    return rows


def get_labeled_reviews(limit: int = 15, offset: int = 0, dataset: str = "indist") -> list:
    return [dict(r) for r in _load(dataset)[offset:offset + limit]]


def get_labeled_stats(dataset: str = "indist") -> dict:
    rows = _load(dataset)
    n = len(rows)
    f = sum(1 for r in rows if r["ground_truth_label"] == 1)
    return {"total": n, "fake": f, "genuine": n - f}
