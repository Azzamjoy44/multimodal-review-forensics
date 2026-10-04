"""
evaluate_kaggle.py
------------------
Evaluates the rule-based scorer and the ML baseline on the Kaggle
fake-reviews dataset (data/fake reviews dataset.csv).

Ground-truth labels in that file:
  CG  (Computer Generated) → suspicious (1)
  OR  (Original / Real)    → genuine    (0)

Rule-based note:
  The near-duplicate check in score_review_list() is O(n²) and assumes a
  single product's reviews — not applicable to a cross-product benchmark.
  Each review is therefore scored independently (no cross-review dup check).
  A stratified sample is used so the evaluation runs in reasonable time.

HOW TO RUN
----------
    python evaluate_kaggle.py              # default 5,000-review sample
    python evaluate_kaggle.py --sample 2000
"""

import argparse
import csv
import os
import random
import warnings

from score_reviews import score_review
from evaluate_against_labels import compute_metrics, SUSPICION_THRESHOLD
from train_ml_baseline import (
    build_pipeline,
    load_kaggle_fake_data,
    safe_fold_count,
)

from sklearn.model_selection import StratifiedKFold, cross_val_predict

KAGGLE_FAKE_FILE = os.path.join(os.path.dirname(__file__), "data", "fake reviews dataset.csv")

DEFAULT_RULE_SAMPLE = 5_000   # reviews used for rule-based evaluation
DEFAULT_ML_SAMPLE   = None    # None = use all 40k rows for ML training


# ---------------------------------------------------------------------------
# Load a balanced sample from the Kaggle dataset for rule-based evaluation
# ---------------------------------------------------------------------------
def load_rule_sample(path, sample_size):
    """
    Load a balanced sample (half CG, half OR) for rule-based evaluation.
    Returns: list of (text, true_label) tuples, where true_label is 0 or 1.
    """
    cg_rows = []
    or_rows = []

    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "").strip().upper()
            text  = row.get("text_", "").strip()
            try:
                rating = int(float(row.get("rating", "3") or "3"))
            except (ValueError, TypeError):
                rating = 3
            entry = (text, rating, label)
            if label == "CG":
                cg_rows.append(entry)
            elif label == "OR":
                or_rows.append(entry)

    half = sample_size // 2
    random.seed(42)
    cg_sample = random.sample(cg_rows, min(half, len(cg_rows)))
    or_sample  = random.sample(or_rows, min(half, len(or_rows)))
    combined   = cg_sample + or_sample
    random.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# Rule-based evaluation on the sample
# ---------------------------------------------------------------------------
def evaluate_rules_on_kaggle(sample):
    """
    Score each review in `sample` independently (no cross-review dup check)
    and compare against the ground-truth CG/OR label.

    Returns metrics dict, y_true, y_pred.
    """
    y_true = []
    y_pred = []

    for text, rating, label in sample:
        review = {"review_id": "k", "review_text": text, "rating": rating}
        result = score_review(review, exact_dups=[], near_dups=[])
        pred   = 1 if result["suspicion_score"] >= SUSPICION_THRESHOLD else 0
        true   = 1 if label == "CG" else 0
        y_true.append(true)
        y_pred.append(pred)

    return compute_metrics(y_true, y_pred), y_true, y_pred


# ---------------------------------------------------------------------------
# ML evaluation on the full (or sampled) Kaggle dataset
# ---------------------------------------------------------------------------
def evaluate_ml_on_kaggle(ml_sample_size=None):
    """
    Train TF-IDF + LR via stratified cross-validation on the Kaggle dataset.
    Returns metrics dict, n_samples, n_folds.
    """
    texts, labels, skipped = load_kaggle_fake_data(sample_size=ml_sample_size)

    if not texts or len(set(labels)) < 2:
        return None, 0, 0

    n_folds  = safe_fold_count(labels, max_folds=5)
    pipeline = build_pipeline()
    cv       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = cross_val_predict(pipeline, texts, labels, cv=cv)

    metrics = compute_metrics(list(labels), list(y_pred))
    return metrics, len(texts), n_folds


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
def print_comparison(rules_m, rules_n, ml_m, ml_n, ml_folds):
    W = 64

    def row(label, rv, mv):
        if isinstance(rv, float) and isinstance(mv, float):
            rs = f"{rv:.1%}" + (" *" if rv > mv else "")
            ms = f"{mv:.1%}" + (" *" if mv > rv else "")
        else:
            rs, ms = str(rv), str(mv)
        print(f"  {label:<14} {rs:^18} {ms:^18}")

    print()
    print("=" * W)
    print("  KAGGLE EVALUATION — Fake Review Detection")
    print("  Dataset: data/fake reviews dataset.csv  (CG=fake / OR=real)")
    print("=" * W)
    print(f"  {'':14} {'Rule-based':^18} {'TF-IDF + LR':^18}")
    print("-" * W)
    row("Reviews used",  rules_n,           ml_n)
    row("CV folds",      "n/a",             ml_folds)
    print("-" * W)
    row("Accuracy",  rules_m["accuracy"],  ml_m["accuracy"])
    row("Precision", rules_m["precision"], ml_m["precision"])
    row("Recall",    rules_m["recall"],    ml_m["recall"])
    row("F1 Score",  rules_m["f1"],        ml_m["f1"])
    print("-" * W)
    print(f"  {'':14} {'Rule-based':^18} {'TF-IDF + LR':^18}")
    print(f"  {'TP (caught)':14} {rules_m['tp']:^18} {ml_m['tp']:^18}")
    print(f"  {'TN (correct)':14} {rules_m['tn']:^18} {ml_m['tn']:^18}")
    print(f"  {'FP (alarm)':14} {rules_m['fp']:^18} {ml_m['fp']:^18}")
    print(f"  {'FN (missed)':14} {rules_m['fn']:^18} {ml_m['fn']:^18}")
    print("=" * W)
    print("  * = better value for that metric")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(rule_sample_size=DEFAULT_RULE_SAMPLE, ml_sample_size=DEFAULT_ML_SAMPLE):
    if not os.path.exists(KAGGLE_FAKE_FILE):
        print(f"Error: '{KAGGLE_FAKE_FILE}' not found.")
        return

    # ---- Rule-based ----
    print(f"Scoring {rule_sample_size:,} sampled reviews with rule-based scorer ...")
    sample       = load_rule_sample(KAGGLE_FAKE_FILE, rule_sample_size)
    rules_m, _, _ = evaluate_rules_on_kaggle(sample)
    rules_n      = len(sample)
    print("Done.")

    # ---- ML ----
    ml_desc = f"{ml_sample_size:,}" if ml_sample_size else "all"
    print(f"\nTraining ML model via cross-validation on {ml_desc} Kaggle reviews ...")
    ml_m, ml_n, ml_folds = evaluate_ml_on_kaggle(ml_sample_size)
    if ml_m is None:
        print("ML evaluation failed.")
        return
    print("Done.")

    # ---- Print results ----
    print_comparison(rules_m, rules_n, ml_m, ml_n, ml_folds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on the Kaggle fake-reviews dataset.")
    parser.add_argument(
        "--sample", type=int, default=DEFAULT_RULE_SAMPLE,
        help=f"Number of reviews for rule-based evaluation (default {DEFAULT_RULE_SAMPLE})",
    )
    args = parser.parse_args()
    main(rule_sample_size=args.sample)
