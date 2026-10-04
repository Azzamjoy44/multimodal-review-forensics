# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
evaluate_against_labels.py
--------------------------
Compares the rule-based scorer's predicted labels against your manual labels
and prints precision, recall, F1, and accuracy.

HOW TO USE
----------
1. Open data/labeled_reviews.csv in Excel (or any editor).
2. For each row, fill in the 'manual_label' column with one of:
     genuine     — you believe the review is real and honest
     suspicious  — you think the review looks fake or manipulated
     uncertain   — you are not sure (these rows are skipped automatically)
3. Leave 'manual_label' blank for rows you haven't annotated yet (also skipped).
4. Save the file, then run:
     python evaluate_against_labels.py

HOW LABELS MAP TO BINARY CLASSIFICATION
-----------------------------------------
  Predicted:  suspicion_score < 45   → 0 (not suspicious)
              suspicion_score >= 45  → 1 (suspicious)
              Recomputed on-the-fly by score_reviews.py —
              the predicted_label column in the CSV is ignored.
              Threshold of 45 was selected via threshold sweep in
              analyze_rule_thresholds.py as the value that maximises F1.

  Manual:     genuine          → 0 (not suspicious)
              suspicious       → 1 (suspicious)
              blank / uncertain → skipped
"""

import csv
import os

from score_reviews import score_review_list

LABELED_FILE = os.path.join(os.path.dirname(__file__), "data", "labeled_reviews.csv")

# Valid manual labels and their binary meaning
#   0 = genuine / not suspicious
#   1 = suspicious / fake
LABEL_TO_BINARY = {
    "genuine":    0,
    "suspicious": 1,
    "fake":       1,   # accepted as a synonym for suspicious
}

# Suspicion score threshold: the cut-off that converts a raw suspicion score
# into a binary prediction (0 = genuine, 1 = suspicious).
#
# Value of 45 was selected by running analyze_rule_thresholds.py and choosing
# the threshold that produced the highest F1 score on the labeled dataset.
# All evaluation scripts import this constant so the threshold is defined
# in exactly one place — change it here and every script updates automatically.
SUSPICION_THRESHOLD = 45


def load_labeled_csv(path):
    """
    Read the labeled CSV and return a list of row dicts.
    """
    if not os.path.exists(path):
        print(f"Error: Could not find '{path}'.")
        print("Make sure you have run evaluate_rules.py first to generate scored data,")
        print("and that data/labeled_reviews.csv exists.")
        return []

    rows = []
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _compute_rule_predictions(rows):
    """
    Run the rule-based scorer on all rows and return a dict mapping
    review_id → binary prediction (0 = genuine, 1 = suspicious).

    This replaces reading predicted_label from the CSV, so the evaluation
    works correctly regardless of whether that column exists or is populated.
    """
    review_dicts = [
        {
            "review_id":   row.get("review_id", ""),
            "review_text": row.get("review_text", ""),
            "rating":      int(row.get("rating", 5) or 5),
        }
        for row in rows
    ]
    scored = score_review_list(review_dicts)
    return {
        r["review_id"]: (1 if r["suspicion_score"] >= SUSPICION_THRESHOLD else 0)
        for r in scored
    }


def convert_to_binary(rows):
    """
    Filter out unlabeled / uncertain rows, then convert labels to 0 or 1.

    Rule-based predictions are computed on-the-fly by the scorer —
    the predicted_label column in the CSV is not used.

    Returns:
      y_true  — list of int (manual labels as binary)
      y_pred  — list of int (rule-based predictions as binary)
      skipped — number of rows that were skipped
    """
    rule_predictions = _compute_rule_predictions(rows)

    y_true  = []
    y_pred  = []
    skipped = 0

    for row in rows:
        manual = row.get("manual_label", "").strip().lower()

        # Skip blank or uncertain rows
        if not manual or manual == "uncertain":
            skipped += 1
            continue

        # Skip rows with an unrecognised manual label and warn
        if manual not in LABEL_TO_BINARY:
            print(f"Warning: Unrecognised label '{manual}' on row {row.get('review_id')} — skipped.")
            skipped += 1
            continue

        rid = row.get("review_id", "")
        y_true.append(LABEL_TO_BINARY[manual])
        y_pred.append(rule_predictions.get(rid, 0))

    return y_true, y_pred, skipped


def compute_metrics(y_true, y_pred):
    """
    Calculate accuracy, precision, recall, and F1 from scratch.
    No external libraries used — just basic counting.

    In our setup:
      Positive class (1) = suspicious review
      Negative class (0) = genuine review

    TP = predicted suspicious, actually suspicious
    FP = predicted suspicious, actually genuine  (false alarm)
    FN = predicted genuine,    actually suspicious (missed catch)
    TN = predicted genuine,    actually genuine
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    total = len(y_true)

    accuracy  = (tp + tn) / total if total > 0 else 0.0

    # Precision: of all reviews flagged as suspicious, how many actually were?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall: of all truly suspicious reviews, how many did we catch?
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1: harmonic mean of precision and recall
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total,
    }


def print_report(metrics, skipped):
    """Print a clear evaluation summary to the terminal."""
    print()
    print("=" * 55)
    print("EVALUATION REPORT — Rule-based scorer vs. manual labels")
    print("=" * 55)
    print(f"  Labeled rows used  : {metrics['total']}")
    print(f"  Rows skipped       : {skipped}  (blank or 'uncertain')")
    print()
    print("  Confusion matrix")
    print(f"    True Positives  (caught suspicious) : {metrics['tp']}")
    print(f"    True Negatives  (correct genuine)   : {metrics['tn']}")
    print(f"    False Positives (false alarm)        : {metrics['fp']}")
    print(f"    False Negatives (missed suspicious)  : {metrics['fn']}")
    print()
    print("  Metrics")
    print(f"    Accuracy  : {metrics['accuracy']:.1%}")
    print(f"    Precision : {metrics['precision']:.1%}")
    print(f"    Recall    : {metrics['recall']:.1%}")
    print(f"    F1 Score  : {metrics['f1']:.1%}")
    print("=" * 55)
    print()

    # Plain-English interpretation
    if metrics["total"] == 0:
        print("No labeled rows to evaluate yet — fill in manual_label in the CSV and re-run.")
    else:
        if metrics["recall"] < 0.5:
            print("Observation: Recall is low — the scorer misses many suspicious reviews.")
        if metrics["precision"] < 0.5:
            print("Observation: Precision is low — the scorer flags many genuine reviews.")
        if metrics["f1"] >= 0.7:
            print("Observation: F1 is reasonable for a rule-based baseline.")


if __name__ == "__main__":
    print(f"Loading: {LABELED_FILE}")

    rows = load_labeled_csv(LABELED_FILE)
    if not rows:
        exit(1)

    print(f"Total rows in file: {len(rows)}")

    y_true, y_pred, skipped = convert_to_binary(rows)

    if len(y_true) == 0:
        print("\nNo annotated rows found yet.")
        print("Open data/labeled_reviews.csv, fill in the 'manual_label' column,")
        print("then re-run this script.")
        exit(0)

    metrics = compute_metrics(y_true, y_pred)
    print_report(metrics, skipped)
