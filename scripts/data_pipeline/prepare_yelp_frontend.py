# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
Run this script once to build data/yelp_frontend_reviews.csv.

It cross-references the yelp_split.csv test split against the three raw Yelp
datasets, keeping only reviews that appear in the test split. The result is a
flat CSV with business_id, rating, date, etc. restored from the raw files.

    python prepare_yelp_frontend.py

Output: data/yelp_frontend_reviews.csv
"""
import os
import re
import csv
from datetime import datetime

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
SPLIT_CSV = os.path.join(DATA_DIR, "yelp_split.csv")
OUT_CSV   = os.path.join(DATA_DIR, "yelp_frontend_reviews.csv")

YELPZIP_META     = os.path.join(DATA_DIR, "YelpZip-20260521T184341Z-3-001", "YelpZip", "metadata")
YELPZIP_REV      = os.path.join(DATA_DIR, "YelpZip-20260521T184341Z-3-001", "YelpZip", "reviewContent")
YELPNYC_META     = os.path.join(DATA_DIR, "YelpNYC-20260521T184345Z-3-001", "YelpNYC", "metadata")
YELPNYC_REV      = os.path.join(DATA_DIR, "YelpNYC-20260521T184345Z-3-001", "YelpNYC", "reviewContent")
YELPCHI_RES_META = os.path.join(DATA_DIR, "YelpChi-20260521T184346Z-3-001", "YelpChi", "output_meta_yelpResData_NRYRcleaned.txt")
YELPCHI_RES_REV  = os.path.join(DATA_DIR, "YelpChi-20260521T184346Z-3-001", "YelpChi", "output_review_yelpResData_NRYRcleaned.txt")
YELPCHI_HOT_META = os.path.join(DATA_DIR, "YelpChi-20260521T184346Z-3-001", "YelpChi", "output_meta_yelpHotelData_NRYRcleaned.txt")
YELPCHI_HOT_REV  = os.path.join(DATA_DIR, "YelpChi-20260521T184346Z-3-001", "YelpChi", "output_review_yelpHotelData_NRYRcleaned.txt")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_test_lookup() -> dict:
    """normalized_text → label for all test-split rows in yelp_split.csv."""
    lookup = {}
    with open(SPLIT_CSV, encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "test":
                lookup[normalize(row["text"])] = int(row["label"])
    print(f"Test split: {len(lookup):,} unique texts.")
    return lookup


def scan_yelpzip(meta_path, review_path, prefix, test_lookup, writer, count, seen):
    label_map = {"-1": 1, "1": 0}
    found = 0
    with open(meta_path, encoding="utf-8", errors="ignore") as mf, \
         open(review_path, encoding="utf-8", errors="ignore") as rf:
        for meta_line, review_line in zip(mf, rf):
            m = meta_line.strip().split("\t")
            r = review_line.strip().split("\t")
            if len(m) < 5 or len(r) < 4:
                continue
            text = "\t".join(r[3:]).strip()
            if not text:
                continue
            norm = normalize(text)
            if norm not in test_lookup or norm in seen:
                continue
            seen.add(norm)
            user_id     = m[0].strip()
            business_id = m[1].strip()
            rating_str  = m[2].strip()
            label_str   = m[3].strip()
            date_str    = m[4].strip()
            try:
                ts = int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())
            except ValueError:
                ts = 0
            writer.writerow({
                "review_id":          f"{prefix}_{business_id}_{count[0]}",
                "review_text":        text,
                "rating":             rating_str,
                "user_id":            user_id,
                "time":               ts,
                "business_id":        f"{prefix}_{business_id}",
                "ground_truth_label": test_lookup.get(norm, label_map.get(label_str, "")),
                "source":             prefix,
            })
            count[0] += 1
            found += 1
    return found


def scan_yelpchi(meta_path, review_path, prefix, test_lookup, writer, count, seen):
    label_map = {"Y": 1, "N": 0}
    found = 0
    with open(meta_path, encoding="utf-8", errors="ignore") as mf, \
         open(review_path, encoding="utf-8", errors="ignore") as rf:
        for meta_line, review_line in zip(mf, rf):
            parts = meta_line.strip().split()
            text  = review_line.strip()
            if len(parts) < 9 or not text:
                continue
            norm = normalize(text)
            if norm not in test_lookup or norm in seen:
                continue
            seen.add(norm)
            date_str    = parts[0]
            user_id     = parts[1]
            business_id = parts[3]
            label_str   = parts[4]
            rating_str  = parts[8]
            try:
                ts = int(datetime.strptime(date_str, "%m/%d/%Y").timestamp())
            except ValueError:
                ts = 0
            writer.writerow({
                "review_id":          f"{prefix}_{business_id}_{count[0]}",
                "review_text":        text,
                "rating":             rating_str,
                "user_id":            user_id,
                "time":               ts,
                "business_id":        f"{prefix}_{business_id}",
                "ground_truth_label": test_lookup.get(norm, label_map.get(label_str, "")),
                "source":             prefix,
            })
            count[0] += 1
            found += 1
    return found


def main():
    print("Loading test split lookup...")
    test_lookup = load_test_lookup()

    fields = ["review_id", "review_text", "rating", "user_id",
              "time", "business_id", "ground_truth_label", "source"]

    count = [0]  # mutable counter shared across scan functions
    seen  = set()  # normalized texts already written — prevents cross-dataset duplicates
    total = 0

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        datasets = [
            (YELPZIP_META,     YELPZIP_REV,      "zip",     scan_yelpzip,  "YelpZip"),
            (YELPNYC_META,     YELPNYC_REV,      "nyc",     scan_yelpzip,  "YelpNYC"),
            (YELPCHI_RES_META, YELPCHI_RES_REV,  "chi_res", scan_yelpchi,  "YelpChi (restaurants)"),
            (YELPCHI_HOT_META, YELPCHI_HOT_REV,  "chi_hot", scan_yelpchi,  "YelpChi (hotels)"),
        ]

        for meta, rev, prefix, scanner, label in datasets:
            if os.path.exists(meta):
                n = scanner(meta, rev, prefix, test_lookup, writer, count, seen)
                print(f"  {label}: {n:,} test reviews found.")
                total += n
            else:
                print(f"  {label}: file not found, skipping.")

    n_fake    = 0
    n_genuine = 0
    with open(OUT_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lbl = row.get("ground_truth_label", "")
            if lbl == "1":
                n_fake += 1
            elif lbl == "0":
                n_genuine += 1

    print(f"\nWrote {total:,} reviews to {OUT_CSV}")
    print(f"  Fake   : {n_fake:,}")
    print(f"  Genuine: {n_genuine:,}")
    print(f"  Total  : {n_fake + n_genuine:,}")


if __name__ == "__main__":
    main()
