# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
export_amazon_results.py
------------------------
Loads the top 100 Amazon reviews (ranked by helpfulness) from data/Reviews.csv,
scores them with the rule-based system and all 8 fake-detection and 8 sentiment
ML models, then writes the full results to data/amazon_top100_results.csv.

HOW TO RUN
----------
    python export_amazon_results.py

OUTPUT
------
    data/amazon_top100_results.csv
"""

import csv
import os
import re

from score_reviews import score_review_list

REVIEWS_FILE = os.path.join(os.path.dirname(__file__), "data", "Reviews.csv")
OUTPUT_FILE  = os.path.join(os.path.dirname(__file__), "data", "amazon_top100_results.csv")

PRODUCT_ID  = "B007JFMH8M"
MODEL_NAMES = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]


def _strip_html(text):
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_top100(product_id=PRODUCT_ID):
    """Return the top 100 reviews for the given product, ranked by helpfulness."""
    rows = []
    with open(REVIEWS_FILE, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("ProductId", "").strip() != product_id:
                continue
            try:
                score = int(float(row.get("Score", "3") or "3"))
            except (ValueError, TypeError):
                score = 3
            rows.append({
                "review_id":   row.get("Id", "").strip(),
                "review_text": _strip_html(row.get("Text", "")),
                "rating":      score,
            })
            if len(rows) == 100:
                break

    return rows


def build_csv_row(review, result):
    """Flatten a scored review into a dict suitable for csv.DictWriter."""
    row = {
        "review_id":    result["review_id"],
        "review_text":  result["review_text"],
        "rating":       result["rating"],
        "rule_score":   result["suspicion_score"],
        "rule_reasons": " | ".join(result.get("rule_reasons", [])),
    }

    for name in MODEL_NAMES:
        # Fake detection
        fd = (result.get("fake_models") or {}).get(name, {})
        label      = fd.get("label")
        confidence = fd.get("confidence")
        reasons    = fd.get("reasons") or []
        row[f"fake_{name}_label"]      = ("fake" if label == 1 else "genuine") if label is not None else "N/A"
        row[f"fake_{name}_confidence"] = f"{confidence}%" if confidence is not None else "N/A"
        row[f"fake_{name}_reasons"]    = " | ".join(reasons)

        # Sentiment
        sd = (result.get("sentiment_models") or {}).get(name, {})
        label      = sd.get("label")
        confidence = sd.get("confidence")
        reasons    = sd.get("reasons") or []
        row[f"sentiment_{name}_label"]      = ("positive" if label == 1 else "negative") if label is not None else "N/A"
        row[f"sentiment_{name}_confidence"] = f"{confidence}%" if confidence is not None else "N/A"
        row[f"sentiment_{name}_reasons"]    = " | ".join(reasons)

    return row


def build_fieldnames():
    fields = ["review_id", "review_text", "rating", "rule_score", "rule_reasons"]
    for name in MODEL_NAMES:
        fields += [
            f"fake_{name}_label",
            f"fake_{name}_confidence",
            f"fake_{name}_reasons",
        ]
    for name in MODEL_NAMES:
        fields += [
            f"sentiment_{name}_label",
            f"sentiment_{name}_confidence",
            f"sentiment_{name}_reasons",
        ]
    return fields


def main():
    print(f"Loading top 100 reviews for product {PRODUCT_ID} from: {REVIEWS_FILE}")
    reviews = load_top100()
    print(f"Loaded {len(reviews)} reviews. Scoring with all models (this may take a moment) ...")

    results = score_review_list(reviews)

    fieldnames = build_fieldnames()
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for review, result in zip(reviews, results):
            writer.writerow(build_csv_row(review, result))

    print(f"\nSaved {len(results)} rows to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
