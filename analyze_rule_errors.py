"""
analyze_rule_errors.py
-----------------------
Loads the manually labeled dataset, recomputes rule-based predictions
on-the-fly, and breaks the results down into the four confusion-matrix
categories: TP, TN, FP, FN.

Saves one CSV per category so you can inspect mistakes in Excel, and
prints a pattern analysis that shows which scoring rules are most
responsible for the errors.

HOW TO RUN
----------
    python analyze_rule_errors.py

OUTPUT FILES
------------
    data/rule_true_positives.csv   — correctly caught suspicious reviews
    data/rule_true_negatives.csv   — correctly left as genuine
    data/rule_false_positives.csv  — genuine reviews wrongly flagged (false alarms)
    data/rule_false_negatives.csv  — suspicious reviews the scorer missed

Each file contains:
    review_id, product_name, review_text, rating,
    manual_label, predicted_score, predicted_label, reasons
"""

import collections
import csv
import os

from score_reviews import score_review_list
from evaluate_against_labels import SUSPICION_THRESHOLD

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(__file__)
LABELED_FILE = os.path.join(BASE_DIR, "data", "labeled_reviews.csv")

OUT_TP = os.path.join(BASE_DIR, "data", "rule_true_positives.csv")
OUT_TN = os.path.join(BASE_DIR, "data", "rule_true_negatives.csv")
OUT_FP = os.path.join(BASE_DIR, "data", "rule_false_positives.csv")
OUT_FN = os.path.join(BASE_DIR, "data", "rule_false_negatives.csv")

# Column order for the output CSVs
OUT_COLUMNS = [
    "review_id", "product_name", "review_text", "rating",
    "manual_label", "predicted_score", "predicted_label", "reasons",
]

# Labels the annotator used — same definitions as evaluate_against_labels.py
LABEL_TO_BINARY = {
    "genuine":    0,
    "suspicious": 1,
    "fake":       1,
}

# SUSPICION_THRESHOLD is imported from evaluate_against_labels.py —
# it is defined there once and shared across all evaluation scripts.
# Change it there and this script updates automatically.


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
# Step 2 — Score all rows with the rule-based scorer
# ---------------------------------------------------------------------------

def score_rows(rows):
    """
    Build minimal review dicts and run them through score_review_list.
    Returns a dict mapping review_id → full scored result
    (suspicion_score, reasons, predicted_label).
    """
    review_dicts = [
        {
            "review_id":   row.get("review_id", ""),
            "review_text": row.get("review_text", ""),
            # Rating is stored as a string in the CSV — cast to int for the scorer
            "rating":      int(row.get("rating", 5) or 5),
        }
        for row in rows
    ]

    scored = score_review_list(review_dicts)

    lookup = {}
    for r in scored:
        score = r["suspicion_score"]
        if score >= 60:
            label = "HIGH"
        elif score >= SUSPICION_THRESHOLD:
            label = "MEDIUM"
        else:
            label = "LOW"

        lookup[r["review_id"]] = {
            "predicted_score": score,
            "predicted_label": label,
            # Store reasons as a readable semicolon-separated string for the CSV
            "reasons": "; ".join(r["reasons"]),
            # Keep reasons as a list for the pattern analysis below
            "reasons_list": r["reasons"],
        }

    return lookup


# ---------------------------------------------------------------------------
# Step 3 — Classify each labeled row into TP / TN / FP / FN
# ---------------------------------------------------------------------------

def classify_rows(rows, score_lookup):
    """
    Compare the manual label against the rule-based prediction for every
    labeled row (blank / uncertain rows are skipped).

    Returns four lists of enriched row dicts: tp, tn, fp, fn.
    Also returns a count of skipped rows.
    """
    tp, tn, fp, fn = [], [], [], []
    skipped = 0

    for row in rows:
        manual = row.get("manual_label", "").strip().lower()

        # Skip blank or uncertain rows
        if not manual or manual == "uncertain":
            skipped += 1
            continue

        if manual not in LABEL_TO_BINARY:
            print(f"  Warning: unknown label '{manual}' on {row.get('review_id')} — skipped.")
            skipped += 1
            continue

        rid        = row.get("review_id", "")
        scored     = score_lookup.get(rid, {})
        pred_score = scored.get("predicted_score", 0)
        pred_label = scored.get("predicted_label", "LOW")
        reasons    = scored.get("reasons", "")

        y_true = LABEL_TO_BINARY[manual]
        y_pred = 1 if pred_score >= SUSPICION_THRESHOLD else 0

        # Build the enriched output row
        enriched = {
            "review_id":       rid,
            "product_name":    row.get("product_name", ""),
            "review_text":     row.get("review_text", ""),
            "rating":          row.get("rating", ""),
            "manual_label":    manual,
            "predicted_score": pred_score,
            "predicted_label": pred_label,
            "reasons":         reasons,
            # Keep the raw list for pattern analysis (not written to CSV)
            "_reasons_list":   scored.get("reasons_list", []),
        }

        if y_true == 1 and y_pred == 1:
            tp.append(enriched)
        elif y_true == 0 and y_pred == 0:
            tn.append(enriched)
        elif y_true == 0 and y_pred == 1:
            fp.append(enriched)   # false alarm: genuine flagged as suspicious
        else:
            fn.append(enriched)   # missed: suspicious labeled as genuine

    return tp, tn, fp, fn, skipped


# ---------------------------------------------------------------------------
# Step 4 — Write one CSV file per category
# ---------------------------------------------------------------------------

def write_category_csv(path, rows):
    """Write a list of enriched rows to a CSV, using OUT_COLUMNS only."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Step 5 — Pattern analysis
# Count how often each scoring reason appears in FP and FN rows.
# This pinpoints which rules are too aggressive (FP) or miss suspicious
# reviews (FN).
# ---------------------------------------------------------------------------

def reason_counts(rows):
    """
    Count how many times each rule fired across the given rows.
    Returns a Counter of short reason prefixes (first ~45 characters),
    which is enough to identify the rule without cluttering the output.
    """
    counter = collections.Counter()
    for row in rows:
        for reason in row.get("_reasons_list", []):
            # Truncate long reasons (e.g. duplicate lists) for cleaner display
            short = reason[:60] + "..." if len(reason) > 60 else reason
            counter[short] += 1
    return counter


def print_pattern_analysis(fp_rows, fn_rows):
    """
    Print which rules fire most often in false positives and false negatives,
    and generate plain-English suggestions for improving the scorer.
    """
    W = 62

    # --- False positive patterns ---
    print()
    print("─" * W)
    print("  FALSE POSITIVE PATTERNS  (genuine reviews wrongly flagged)")
    print("─" * W)
    fp_counts = reason_counts(fp_rows)
    if fp_counts:
        for reason, count in fp_counts.most_common():
            print(f"    {count:>2}x  {reason}")
    else:
        print("    (none)")

    # --- False negative patterns ---
    print()
    print("─" * W)
    print("  FALSE NEGATIVE PATTERNS  (suspicious reviews missed)")
    print("─" * W)
    fn_counts = reason_counts(fn_rows)
    if fn_counts:
        for reason, count in fn_counts.most_common():
            print(f"    {count:>2}x  {reason}")
    else:
        print("    (none)")

    # --- Suggestions ---
    print()
    print("─" * W)
    print("  SUGGESTIONS")
    print("─" * W)

    # Rules that appear frequently in FPs are too aggressive
    aggressive_rules = [r for r, c in fp_counts.most_common(3) if c >= 2]
    if aggressive_rules:
        print("  Rules that may be too aggressive (appear often in false positives):")
        for r in aggressive_rules:
            print(f"    - {r}")
        print()
        print("  Consider: raising the score threshold for these rules, or")
        print("  adding an exception (e.g. short reviews with a specific product")
        print("  mention should not be penalised as heavily).")
    else:
        print("  No single rule dominates false positives — precision issues")
        print("  may be caused by combinations of mild signals rather than one")
        print("  aggressive rule.")

    print()

    # If false negatives are all genuinely short/generic, suggest new rules
    if fn_rows:
        avg_fn_len = sum(len(r["review_text"]) for r in fn_rows) / len(fn_rows)
        print(f"  Missed suspicious reviews average text length: {avg_fn_len:.0f} chars")
        if avg_fn_len > 100:
            print("  Observation: missed reviews tend to be longer — the scorer")
            print("  currently focuses heavily on short reviews. Consider adding")
            print("  rules that look at content quality in longer reviews, e.g.")
            print("  vague praise with no specific game details mentioned.")
        else:
            print("  Observation: missed reviews are short but were not caught —")
            print("  they may lack the promotional words the scorer looks for.")
            print("  Consider expanding the promotional word list.")

    print()
    print("  General next steps:")
    print("  1. Open data/rule_false_positives.csv and read the 'reasons' column.")
    print("     Identify which rule caused the false alarm and decide if the")
    print("     threshold for that rule should be raised.")
    print("  2. Open data/rule_false_negatives.csv and read the review texts.")
    print("     Look for patterns the current rules do not cover (e.g. vague")
    print("     one-liners that still pass as genuine, or reviews in other")
    print("     languages that slip through).")
    print("  3. Edit score_reviews.py to adjust thresholds or add new rules.")
    print("  4. Re-run: python evaluate_against_labels.py")
    print("             python compare_models.py")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 62)
    print("analyze_rule_errors.py")
    print("=" * 62)

    # Step 1: load
    print(f"\nLoading: {LABELED_FILE}")
    rows = load_csv(LABELED_FILE)
    if not rows:
        exit(1)
    print(f"  Total rows in file: {len(rows)}")

    # Step 2: score
    print("\nScoring with rule-based scorer...")
    score_lookup = score_rows(rows)

    # Step 3: classify
    tp, tn, fp, fn, skipped = classify_rows(rows, score_lookup)
    labeled = len(tp) + len(tn) + len(fp) + len(fn)

    # Step 4: write CSVs
    write_category_csv(OUT_TP, tp)
    write_category_csv(OUT_TN, tn)
    write_category_csv(OUT_FP, fp)
    write_category_csv(OUT_FN, fn)

    # Summary table
    print()
    print("=" * 62)
    print("  CONFUSION MATRIX SUMMARY")
    print("=" * 62)
    print(f"  Labeled rows used : {labeled}")
    print(f"  Rows skipped      : {skipped}  (blank or 'uncertain')")
    print()
    print(f"  True  Positives (TP) — correctly caught suspicious : {len(tp):>3}")
    print(f"  True  Negatives (TN) — correctly left as genuine   : {len(tn):>3}")
    print(f"  False Positives (FP) — genuine wrongly flagged     : {len(fp):>3}  ← precision problem")
    print(f"  False Negatives (FN) — suspicious reviews missed   : {len(fn):>3}  ← recall problem")
    print()

    total = labeled
    accuracy  = (len(tp) + len(tn)) / total if total else 0
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) else 0
    recall    = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0)

    print(f"  Accuracy  : {accuracy:.1%}")
    print(f"  Precision : {precision:.1%}")
    print(f"  Recall    : {recall:.1%}")
    print(f"  F1 Score  : {f1:.1%}")
    print()
    print(f"  Output files written:")
    print(f"    {OUT_TP}")
    print(f"    {OUT_TN}")
    print(f"    {OUT_FP}")
    print(f"    {OUT_FN}")
    print("=" * 62)

    # Step 5: pattern analysis and suggestions
    print_pattern_analysis(fp, fn)
