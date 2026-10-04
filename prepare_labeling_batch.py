"""
prepare_labeling_batch.py
--------------------------
Fetches reviews from several Steam games, scores them with the rule-based
scorer, and writes two CSV files ready for manual labeling.

NOTE: Cyberpunk 2077 (app_id 1091500) is intentionally excluded here
because it is already the source of most rows in labeled_reviews.csv.
Using different games increases product diversity in the labeled dataset.

OUTPUT FILES
------------
  data/labeling_batch_internal.csv
      Full detail including predicted_score and predicted_label.
      Keep this for your own reference and thesis analysis — compare it
      against your manual labels AFTER you have finished annotating.

  data/labeling_batch_blind.csv
      The file you should open in Excel to annotate.
      predicted_score and predicted_label are removed so the model's
      guesses do not influence your manual labeling decisions.

SAMPLING STRATEGY (thesis-explainable)
---------------------------------------
After scoring, reviews are divided into three risk tiers:
    HIGH   (score >= 60)   — strong suspicious signals
    MEDIUM (30 <= score < 60) — moderate signals
    LOW    (score < 30)    — likely genuine

We draw an equal share from each tier instead of taking reviews randomly.
This guarantees that the labeling batch contains a meaningful number of
suspicious-looking reviews, which is the current bottleneck for the ML model.
You can explain this in your thesis as "stratified sampling by predicted
risk tier to ensure class balance in the labeling batch."

HOW TO RUN
----------
    python prepare_labeling_batch.py

AFTER LABELING
--------------
1.  Open data/labeling_batch_blind.csv in Excel.
2.  For each row, fill in the 'manual_label' column with one of:
        genuine     — the review looks real and honest
        suspicious  — the review looks fake or manipulated
        uncertain   — you cannot decide (rows are skipped automatically)
3.  Save the file as CSV (keep UTF-8 encoding).
4.  Append the labeled rows to data/labeled_reviews.csv
    (paste the data rows below the existing ones — do NOT copy the header again).
5.  Re-run the evaluation pipeline:
        python evaluate_against_labels.py
        python train_ml_baseline.py
        python compare_models.py
"""

import csv
import os

from fetch_steam_reviews import fetch_steam_reviews
from score_reviews import score_review_list

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Steam app IDs to fetch reviews from.
# Cyberpunk 2077 (1091500) is intentionally excluded — it already dominates
# data/labeled_reviews.csv and adding more of the same game would reduce
# the diversity of the labeled dataset.
#
#   570     = Dota 2
#   730     = Counter-Strike 2
#   292030  = The Witcher 3: Wild Hunt
#   440     = Team Fortress 2
#   304930  = Unturned
#   1174180 = Red Dead Redemption 2
APP_IDS = [570, 730, 292030, 440, 304930, 1174180]

# How many reviews to request from Steam per app (max 100 per API call).
REVIEWS_PER_APP = 30

# Total reviews to keep in the final batch after deduplication and sampling.
# Roughly one third will be HIGH-risk, one third MEDIUM, one third LOW.
BATCH_SIZE = 120

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(__file__)
LABELED_FILE  = os.path.join(BASE_DIR, "data", "labeled_reviews.csv")
INTERNAL_FILE = os.path.join(BASE_DIR, "data", "labeling_batch_internal.csv")
BLIND_FILE    = os.path.join(BASE_DIR, "data", "labeling_batch_blind.csv")

# Column order for each output file
INTERNAL_COLUMNS = [
    "review_id", "product_id", "product_name", "review_text",
    "rating", "review_date", "reviewer_id", "source",
    "predicted_score", "predicted_label", "notes",
]

BLIND_COLUMNS = [
    "review_id", "product_name", "review_text",
    "rating", "manual_label", "notes",
]

# Score thresholds — must stay in sync with score_reviews.py
HIGH_THRESHOLD   = 60   # score >= 60  →  HIGH
MEDIUM_THRESHOLD = 30   # score >= 30  →  MEDIUM  |  score < 30  →  LOW


# ---------------------------------------------------------------------------
# Helper: load review_ids already present in labeled_reviews.csv
# ---------------------------------------------------------------------------

def load_existing_ids(path):
    """
    Return the set of review_ids that are already in the working labeled CSV.
    This prevents the labeling batch from containing rows we have already seen.
    Returns an empty set if the file does not exist yet.
    """
    if not os.path.exists(path):
        return set()
    existing = set()
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("review_id", "").strip()
            if rid:
                existing.add(rid)
    return existing


# ---------------------------------------------------------------------------
# Step 1 — Fetch reviews from every configured app ID
# ---------------------------------------------------------------------------

def fetch_all_reviews():
    """
    Call the Steam fetcher for each app ID and return one combined list.
    Prints progress so you can see what is happening.
    """
    all_reviews = []
    for app_id in APP_IDS:
        print(f"  Fetching {REVIEWS_PER_APP} reviews for app_id={app_id}...")
        reviews = fetch_steam_reviews(app_id, count=REVIEWS_PER_APP)
        if reviews:
            print(f"    Received {len(reviews)} reviews.")
            all_reviews.extend(reviews)
        else:
            print(f"    No reviews returned — skipping app_id={app_id}.")
    return all_reviews


# ---------------------------------------------------------------------------
# Step 2 — Remove duplicates and already-labeled reviews
# ---------------------------------------------------------------------------

def deduplicate(reviews, existing_ids):
    """
    Keep only reviews whose review_id:
      - has not appeared earlier in this same fetch (intra-batch duplicates)
      - is not already present in data/labeled_reviews.csv

    Returns (unique_list, n_intra_removed, n_already_labeled).
    """
    seen_this_batch = set()
    unique          = []
    n_intra         = 0
    n_preexisting   = 0

    for r in reviews:
        rid = r["review_id"]

        if rid in existing_ids:
            n_preexisting += 1
            continue

        if rid in seen_this_batch:
            n_intra += 1
            continue

        seen_this_batch.add(rid)
        unique.append(r)

    return unique, n_intra, n_preexisting


# ---------------------------------------------------------------------------
# Step 3 — Score reviews and attach predicted_score / predicted_label
# ---------------------------------------------------------------------------

def score_and_attach(reviews):
    """
    Run the rule-based scorer on all reviews at once (one batch pass so that
    near-duplicate detection works across the whole set) and return an
    enriched list with 'predicted_score' and 'predicted_label' added.
    """
    scored = score_review_list(reviews)

    # Build lookup: review_id → scored result
    score_lookup = {r["review_id"]: r for r in scored}

    enriched = []
    for review in reviews:
        rid   = review["review_id"]
        s     = score_lookup.get(rid, {})
        score = s.get("suspicion_score", 0)

        if score >= HIGH_THRESHOLD:
            label = "HIGH"
        elif score >= MEDIUM_THRESHOLD:
            label = "MEDIUM"
        else:
            label = "LOW"

        enriched.append({
            **review,                    # all fields from the fetcher
            "predicted_score": score,
            "predicted_label": label,
            "manual_label":    "",       # blank — annotator fills this in
            "notes":           "",       # blank — annotator fills this in
        })

    return enriched


# ---------------------------------------------------------------------------
# Step 4 — Stratified sampling: equal share from each risk tier
# ---------------------------------------------------------------------------

def stratified_sample(enriched, total):
    """
    Select `total` reviews spread across HIGH, MEDIUM, and LOW tiers.

    Target: total // 3 from each tier.
    If a tier is smaller than its target, take everything from it and
    distribute the shortfall proportionally to the remaining tiers.
    Reviews within each tier are taken in the order they were fetched
    (no random shuffling — reproducible and easy to explain).

    Returns reviews sorted HIGH → MEDIUM → LOW.
    """
    high   = [r for r in enriched if r["predicted_label"] == "HIGH"]
    medium = [r for r in enriched if r["predicted_label"] == "MEDIUM"]
    low    = [r for r in enriched if r["predicted_label"] == "LOW"]

    # Ideal per-tier target
    per_tier = total // 3
    extra    = total - per_tier * 3  # 0, 1, or 2 left over — give to HIGH first

    want_high   = per_tier + (1 if extra >= 1 else 0)
    want_medium = per_tier + (1 if extra >= 2 else 0)
    want_low    = per_tier

    # Take up to the target from each tier
    got_high   = high[:want_high]
    got_medium = medium[:want_medium]
    got_low    = low[:want_low]

    # If any tier ran short, fill the gap from the leftover of the others
    shortfall = total - len(got_high) - len(got_medium) - len(got_low)
    if shortfall > 0:
        # Collect unused reviews from all three tiers
        unused = (
            high[want_high:]     +
            medium[want_medium:] +
            low[want_low:]
        )
        got_low = got_low + unused[:shortfall]

    return got_high + got_medium + got_low


# ---------------------------------------------------------------------------
# Step 5 — Write the two output CSV files
# ---------------------------------------------------------------------------

def write_csvs(batch):
    """
    Write the internal file (full columns) and the blind file (no predictions).
    """
    with open(INTERNAL_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INTERNAL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(batch)

    with open(BLIND_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BLIND_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("prepare_labeling_batch.py")
    print("=" * 60)
    print(f"\nGames to fetch (Cyberpunk 2077 excluded):")
    for aid in APP_IDS:
        print(f"  app_id={aid}")

    # Load IDs that are already in the labeled CSV so we can skip them
    existing_ids = load_existing_ids(LABELED_FILE)
    print(f"\nExisting labeled reviews to skip: {len(existing_ids)}")

    # Step 1: fetch
    print("\n[1/4] Fetching reviews from Steam...")
    all_reviews = fetch_all_reviews()
    print(f"\n  Total fetched: {len(all_reviews)}")

    # Step 2: deduplicate
    print("\n[2/4] Removing duplicates and already-labeled reviews...")
    unique, n_intra, n_preexisting = deduplicate(all_reviews, existing_ids)
    print(f"  Intra-batch duplicates removed : {n_intra}")
    print(f"  Already in labeled CSV skipped : {n_preexisting}")
    print(f"  Unique new reviews remaining   : {len(unique)}")

    if len(unique) == 0:
        print("\nNothing new to save. Try increasing REVIEWS_PER_APP.")
        exit(0)

    # Step 3: score
    print("\n[3/4] Scoring with rule-based scorer...")
    enriched = score_and_attach(unique)

    n_high   = sum(1 for r in enriched if r["predicted_label"] == "HIGH")
    n_medium = sum(1 for r in enriched if r["predicted_label"] == "MEDIUM")
    n_low    = sum(1 for r in enriched if r["predicted_label"] == "LOW")
    print(f"  Risk distribution: {n_high} HIGH  /  {n_medium} MEDIUM  /  {n_low} LOW")

    # Step 4: sample
    actual_total = min(BATCH_SIZE, len(enriched))
    print(f"\n[4/4] Sampling {actual_total} reviews (stratified by risk tier)...")
    batch = stratified_sample(enriched, actual_total)

    b_high   = sum(1 for r in batch if r["predicted_label"] == "HIGH")
    b_medium = sum(1 for r in batch if r["predicted_label"] == "MEDIUM")
    b_low    = sum(1 for r in batch if r["predicted_label"] == "LOW")
    print(f"  Batch breakdown: {b_high} HIGH  /  {b_medium} MEDIUM  /  {b_low} LOW")

    # Count how many games are represented
    games_in_batch = {r["product_name"] for r in batch}
    print(f"  Products in batch: {len(games_in_batch)}")
    for g in sorted(games_in_batch):
        n = sum(1 for r in batch if r["product_name"] == g)
        print(f"    {g}: {n} reviews")

    # Step 5: write
    write_csvs(batch)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Reviews fetched              : {len(all_reviews)}")
    print(f"  After deduplication          : {len(unique)}")
    print(f"  Saved to batch               : {len(batch)}")
    print()
    print(f"  Internal file (full detail)  : {INTERNAL_FILE}")
    print(f"  Blind file   (label this)    : {BLIND_FILE}")
    print()
    print("Next steps:")
    print("  1. Open data/labeling_batch_blind.csv in Excel.")
    print("  2. Fill in 'manual_label' for each row:")
    print("       genuine / suspicious / uncertain")
    print("  3. Save as CSV.")
    print("  4. Append the labeled rows to data/labeled_reviews.csv")
    print("     (paste data rows below the existing ones — no duplicate header).")
    print("  5. Run:")
    print("       python evaluate_against_labels.py")
    print("       python train_ml_baseline.py")
    print("       python compare_models.py")
