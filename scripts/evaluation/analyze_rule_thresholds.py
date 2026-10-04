# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
analyze_rule_thresholds.py
---------------------------
Sweeps multiple suspicion-score thresholds and measures how precision,
recall, and F1 change at each one.

The rule-based scorer gives every review a score from 0 to 100.
The threshold is the cut-off that decides: "above this score = suspicious".
Different thresholds produce different precision/recall trade-offs:

  Lower threshold  → flags more reviews as suspicious
                     → higher recall   (catch more real fakes)
                     → lower precision (more false alarms)

  Higher threshold → flags fewer reviews as suspicious
                     → higher precision (fewer false alarms)
                     → lower recall    (miss more real fakes)

This script helps you find the threshold that best fits your thesis goal.

HOW TO RUN
----------
    python analyze_rule_thresholds.py

OUTPUT
------
    data/rule_threshold_analysis.csv
        One row per threshold with all metrics — open in Excel to plot a
        precision/recall curve.
"""

import csv
import os

from score_reviews import score_review_list

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(__file__)
LABELED_FILE = os.path.join(BASE_DIR, "data", "labeled_reviews.csv")
OUTPUT_FILE  = os.path.join(BASE_DIR, "data", "rule_threshold_analysis.csv")

# ---------------------------------------------------------------------------
# Thresholds to evaluate.
# These cover the full range from very permissive (10) to very strict (60).
# ---------------------------------------------------------------------------
THRESHOLDS = [10, 20, 25, 30, 35, 40, 45, 50, 60]

# Manual label → binary mapping (same as evaluate_against_labels.py)
LABEL_TO_BINARY = {
    "genuine":    0,
    "suspicious": 1,
    "fake":       1,
}


# ---------------------------------------------------------------------------
# Step 1 — Load the labeled CSV
# ---------------------------------------------------------------------------

def load_csv(path):
    """Read all rows from the labeled CSV. Returns a list of dicts."""
    if not os.path.exists(path):
        print(f"Error: '{path}' not found.")
        return []
    rows = []
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Step 2 — Score every row with the rule-based scorer and pair each score
#           with its manual binary label, skipping uncertain/blank rows.
# ---------------------------------------------------------------------------

def get_scores_and_labels(rows):
    """
    Run the scorer on all rows and return two parallel lists:
      scores  — rule-based suspicion score (0–100) for each usable row
      y_true  — manual binary label (0 = genuine, 1 = suspicious)

    Also returns the number of skipped rows.
    """
    # Build minimal review dicts for the scorer
    review_dicts = [
        {
            "review_id":   row.get("review_id", ""),
            "review_text": row.get("review_text", ""),
            "rating":      int(row.get("rating", 5) or 5),
        }
        for row in rows
    ]

    # Score everything in one pass (so near-duplicate detection is global)
    scored = score_review_list(review_dicts)
    score_lookup = {r["review_id"]: r["suspicion_score"] for r in scored}

    scores  = []
    y_true  = []
    skipped = 0

    for row in rows:
        manual = row.get("manual_label", "").strip().lower()

        # Skip blank or uncertain rows
        if not manual or manual == "uncertain":
            skipped += 1
            continue

        if manual not in LABEL_TO_BINARY:
            skipped += 1
            continue

        rid = row.get("review_id", "")
        scores.append(score_lookup.get(rid, 0))
        y_true.append(LABEL_TO_BINARY[manual])

    return scores, y_true, skipped


# ---------------------------------------------------------------------------
# Step 3 — Compute metrics at a single threshold
# ---------------------------------------------------------------------------

def metrics_at_threshold(scores, y_true, threshold):
    """
    Convert scores to binary predictions using the given threshold,
    then compute TP/TN/FP/FN and derived metrics.

    Returns a dict with all values.
    """
    y_pred = [1 if s >= threshold else 0 for s in scores]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    total     = len(y_true)
    accuracy  = (tp + tn) / total            if total            else 0.0
    precision = tp / (tp + fp)               if (tp + fp) > 0    else 0.0
    recall    = tp / (tp + fn)               if (tp + fn) > 0    else 0.0
    f1        = (2 * precision * recall /
                 (precision + recall))        if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


# ---------------------------------------------------------------------------
# Step 4 — Print a formatted table of all results
# ---------------------------------------------------------------------------

def print_table(results):
    """Print a clean side-by-side table of metrics at each threshold."""
    W = 74
    print()
    print("=" * W)
    print("  THRESHOLD SWEEP — Rule-based Scorer")
    print("=" * W)
    print(f"  {'Thresh':>6}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}"
          f"  {'TP':>4}  {'TN':>4}  {'FP':>4}  {'FN':>4}")
    print("-" * W)

    best_f1        = max(results, key=lambda r: r["f1"])
    best_precision = max(results, key=lambda r: r["precision"])
    best_recall    = max(results, key=lambda r: r["recall"])

    for r in results:
        # Mark the best row for each metric with a symbol
        tags = []
        if r["threshold"] == best_f1["threshold"]:
            tags.append("← best F1")
        if r["threshold"] == best_precision["threshold"] and r["threshold"] != best_f1["threshold"]:
            tags.append("← best Precision")
        if r["threshold"] == best_recall["threshold"] and r["threshold"] != best_f1["threshold"]:
            tags.append("← best Recall")
        tag = "  " + "  ".join(tags) if tags else ""

        print(f"  {r['threshold']:>6}  "
              f"{r['accuracy']:.1%}  "
              f"{r['precision']:.1%}  "
              f"{r['recall']:.1%}  "
              f"{r['f1']:.1%}  "
              f"{r['tp']:>4}  {r['tn']:>4}  {r['fp']:>4}  {r['fn']:>4}"
              f"{tag}")

    print("=" * W)


# ---------------------------------------------------------------------------
# Step 5 — Print interpretation and threshold recommendation
# ---------------------------------------------------------------------------

def print_interpretation(results):
    """
    Plain-English explanation of the trade-off and a recommendation for
    which threshold to use in the thesis.
    """
    best_f1        = max(results, key=lambda r: r["f1"])
    best_precision = max(results, key=lambda r: r["precision"])
    best_recall    = max(results, key=lambda r: r["recall"])

    W = 74
    print()
    print("─" * W)
    print("  INTERPRETATION")
    print("─" * W)
    print()
    print("  How to read this table:")
    print("  - Lower threshold → more reviews flagged → higher recall,")
    print("    lower precision (more false alarms).")
    print("  - Higher threshold → fewer reviews flagged → higher precision,")
    print("    lower recall (more suspicious reviews missed).")
    print()
    print(f"  Best F1        : threshold = {best_f1['threshold']}"
          f"  (F1 {best_f1['f1']:.1%},"
          f"  Precision {best_f1['precision']:.1%},"
          f"  Recall {best_f1['recall']:.1%})")
    print(f"  Best Precision : threshold = {best_precision['threshold']}"
          f"  (Precision {best_precision['precision']:.1%},"
          f"  Recall {best_precision['recall']:.1%})")
    print(f"  Best Recall    : threshold = {best_recall['threshold']}"
          f"  (Recall {best_recall['recall']:.1%},"
          f"  Precision {best_recall['precision']:.1%})")
    print()
    print("  ── Choosing a threshold for your thesis ──────────────────────")
    print()
    print(f"  For BALANCED detection (best overall performance):")
    print(f"    Use threshold = {best_f1['threshold']}")
    print(f"    This maximises F1 — the harmonic mean of precision and recall.")
    print(f"    It is the standard choice when both false positives and false")
    print(f"    negatives carry roughly equal cost.")
    print()
    print(f"  For CONSERVATIVE alerting (fewer false alarms, higher trust):")
    print(f"    Use threshold = {best_precision['threshold']}")
    print(f"    This maximises precision — you will flag fewer genuine reviews")
    print(f"    by mistake, at the cost of missing more real fakes.")
    print(f"    Appropriate if the system is shown to end users who would lose")
    print(f"    trust from seeing false positives.")
    print()
    print("  Note: The active threshold used by all evaluation scripts is")
    print("  defined as SUSPICION_THRESHOLD in evaluate_against_labels.py.")
    print("  To adopt a different threshold, change that single constant")
    print("  and re-run: evaluate_against_labels.py, compare_models.py,")
    print("  and analyze_rule_errors.py.")
    print()


# ---------------------------------------------------------------------------
# Step 6 — Save results to CSV
# ---------------------------------------------------------------------------

def save_csv(results, path):
    """Write the threshold sweep results to a CSV file."""
    fieldnames = ["threshold", "accuracy", "precision", "recall", "f1",
                  "tp", "tn", "fp", "fn"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 74)
    print("  analyze_rule_thresholds.py")
    print("=" * 74)

    # Load labeled data
    print(f"\nLoading: {LABELED_FILE}")
    rows = load_csv(LABELED_FILE)
    if not rows:
        exit(1)
    print(f"  Total rows in file: {len(rows)}")

    # Score and pair with manual labels
    print("\nScoring with rule-based scorer...")
    scores, y_true, skipped = get_scores_and_labels(rows)
    print(f"  Usable labeled rows : {len(y_true)}")
    print(f"  Skipped             : {skipped}  (blank or 'uncertain')")
    print(f"  Suspicious (manual) : {sum(y_true)}")
    print(f"  Genuine    (manual) : {len(y_true) - sum(y_true)}")

    if len(y_true) == 0:
        print("\nNo labeled rows found. Fill in manual_label in the CSV first.")
        exit(0)

    # Sweep thresholds
    results = [metrics_at_threshold(scores, y_true, t) for t in THRESHOLDS]

    # Print table and interpretation
    print_table(results)
    print_interpretation(results)

    # Save CSV
    save_csv(results, OUTPUT_FILE)
    print(f"  Results saved to: {OUTPUT_FILE}")
    print()
