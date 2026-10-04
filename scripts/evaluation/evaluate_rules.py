# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
evaluate_rules.py
-----------------
Runs the rule-based scoring pipeline on a set of reviews and produces:
  1. A printed evaluation summary in the terminal
  2. data/scored_reviews.json  — full scored results (easy to reload later)
  3. data/scored_reviews.csv   — spreadsheet-friendly format for thesis tables

Supports two data sources (controlled by the SOURCE constant below):
  "sample"  — loads from data/sample_reviews.json (no internet needed)
  "steam"   — fetches live reviews from Steam for a given app ID
"""

import json
import csv
import os

# Re-use existing project modules — no logic duplicated here
from load_reviews import load_reviews
from fetch_steam_reviews import fetch_steam_reviews
from score_reviews import score_review_list

# ---------------------------------------------------------------------------
# Configuration — edit these two lines to switch source or app
# ---------------------------------------------------------------------------
SOURCE = "steam"       # "sample"  or  "steam"
STEAM_APP_ID = 1091500  # Only used when SOURCE = "steam" (Cyberpunk 2077)
STEAM_COUNT  = 50       # How many live reviews to fetch (max 100)

# Output folder — same folder as the existing sample data
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Step 1: Load reviews from the chosen source
# ---------------------------------------------------------------------------
def load_source_reviews():
    """
    Return a plain list of review dicts from whichever source is configured.
    """
    if SOURCE == "steam":
        print(f"Fetching {STEAM_COUNT} live reviews from Steam (app {STEAM_APP_ID})…")
        reviews = fetch_steam_reviews(STEAM_APP_ID, count=STEAM_COUNT)
    else:
        print("Loading reviews from data/sample_reviews.json…")
        reviews = load_reviews()   # returns all reviews in the file

    return reviews


# ---------------------------------------------------------------------------
# Step 2: Print an evaluation summary to the terminal
# ---------------------------------------------------------------------------
def print_summary(scored):
    """
    Print overall stats and the top-5 most suspicious reviews.
    `scored` is the list returned by score_review_list().
    """
    total = len(scored)
    if total == 0:
        print("No reviews to evaluate.")
        return

    # --- Overall stats ---
    avg_score = sum(r["suspicion_score"] for r in scored) / total

    low_count    = sum(1 for r in scored if r["suspicion_score"] < 30)
    medium_count = sum(1 for r in scored if 30 <= r["suspicion_score"] < 60)
    high_count   = sum(1 for r in scored if r["suspicion_score"] >= 60)

    print()
    print("=" * 60)
    print("EVALUATION SUMMARY — Rule-based scorer")
    print("=" * 60)
    print(f"  Total reviews   : {total}")
    print(f"  Avg score       : {avg_score:.1f} / 100")
    print(f"  LOW  (0–29)     : {low_count}")
    print(f"  MEDIUM (30–59)  : {medium_count}")
    print(f"  HIGH (60–100)   : {high_count}")
    print("=" * 60)

    # --- Top 5 most suspicious ---
    top5 = sorted(scored, key=lambda r: r["suspicion_score"], reverse=True)[:5]

    print("\nTOP 5 MOST SUSPICIOUS REVIEWS")
    print("-" * 60)
    for rank, review in enumerate(top5, start=1):
        short_text = review["review_text"][:80].replace("\n", " ")
        if len(review["review_text"]) > 80:
            short_text += "…"

        print(f"\n#{rank}  [{review['review_id']}]  score={review['suspicion_score']}/100"
              f"  rating={review['rating']}/5")
        print(f"     \"{short_text}\"")
        print(f"     Reasons:")
        for reason in review["reasons"]:
            print(f"       - {reason}")

    print()


# ---------------------------------------------------------------------------
# Step 3a: Save results as JSON
# ---------------------------------------------------------------------------
def save_json(scored, path):
    """Write the full scored list to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)
    print(f"JSON saved → {path}")


# ---------------------------------------------------------------------------
# Step 3b: Save results as CSV
# ---------------------------------------------------------------------------
def save_csv(scored, path):
    """
    Write scored results to a CSV file.
    The 'reasons' list is joined into a single string separated by ' | '.
    """
    if not scored:
        return

    fieldnames = ["review_id", "rating", "suspicion_score", "risk_level",
                  "review_text", "reasons"]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in scored:
            score = r["suspicion_score"]
            risk  = "LOW" if score < 30 else "MEDIUM" if score < 60 else "HIGH"

            writer.writerow({
                "review_id":       r["review_id"],
                "rating":          r["rating"],
                "suspicion_score": score,
                "risk_level":      risk,
                "review_text":     r["review_text"],
                "reasons":         " | ".join(r["reasons"]),
            })

    print(f"CSV  saved → {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load reviews
    reviews = load_source_reviews()

    if not reviews:
        print("No reviews loaded. Check your source configuration and try again.")
        exit(1)

    print(f"Loaded {len(reviews)} review(s). Scoring…")

    # 2. Score every review using the shared pipeline
    scored = score_review_list(reviews)

    # 3. Print evaluation summary
    print_summary(scored)

    # 4. Save output files into the data/ folder
    os.makedirs(DATA_DIR, exist_ok=True)
    save_json(scored, os.path.join(DATA_DIR, "scored_reviews.json"))
    save_csv( scored, os.path.join(DATA_DIR, "scored_reviews.csv"))

    print("\nDone.")
