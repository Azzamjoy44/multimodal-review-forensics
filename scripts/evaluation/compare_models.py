# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
compare_models.py
-----------------
Produces a side-by-side thesis comparison of two approaches:

  1. Rule-based scorer  — uses hand-crafted rules (score_reviews.py)
                          Predictions are already stored in the CSV as
                          'predicted_label' (LOW / MEDIUM / HIGH).

  2. ML baseline        — TF-IDF + Logistic Regression (train_ml_baseline.py)
                          Predictions are generated here via cross-validation
                          so they are fair out-of-sample estimates.

Both are evaluated on the SAME labeled rows using the SAME metric definitions:

  Binary label mapping
    genuine          → 0   (not suspicious)
    suspicious/fake  → 1   (suspicious)
    LOW predicted    → 0
    MEDIUM/HIGH      → 1

  Metric focus: SUSPICIOUS class (positive class = 1)
    Precision = TP / (TP + FP)  — of flagged reviews, how many were truly suspicious?
    Recall    = TP / (TP + FN)  — of all suspicious reviews, how many were caught?
    F1        = harmonic mean of Precision and Recall
    Accuracy  = (TP + TN) / total

  Rule-based: predictions come from the 'predicted_label' column in the CSV
              (pre-computed by score_reviews.py, no training needed).

  ML baseline: predictions come from cross_val_predict (stratified k-fold),
               so every sample is scored by a model that never saw it during
               training — giving fair out-of-sample estimates.

  Using the same metric function (compute_metrics from evaluate_against_labels.py)
  for both models guarantees the numbers are directly comparable.

HOW TO RUN
----------
    python compare_models.py
"""

import os
import warnings

from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ---------------------------------------------------------------------------
# Reuse loading, label-conversion, and metric logic from existing scripts
# so we do not duplicate code.
# ---------------------------------------------------------------------------
from evaluate_against_labels import (
    load_labeled_csv,
    convert_to_binary,
    compute_metrics,
)
from train_ml_baseline import (
    build_pipeline,
    load_labeled_data,
    safe_fold_count,
)

LABELED_FILE = os.path.join(os.path.dirname(__file__), "data", "labeled_reviews.csv")


# ---------------------------------------------------------------------------
# Helper: format a metrics dict as a compact one-liner for the table
# ---------------------------------------------------------------------------
def _fmt(m):
    return (
        f"  Accuracy : {m['accuracy']:.1%}\n"
        f"  Precision: {m['precision']:.1%}\n"
        f"  Recall   : {m['recall']:.1%}\n"
        f"  F1       : {m['f1']:.1%}"
    )


# ---------------------------------------------------------------------------
# Rule-based evaluation
# Reads predictions from the 'predicted_label' column already in the CSV.
# LOW → 0 (not suspicious), MEDIUM / HIGH → 1 (suspicious).
# ---------------------------------------------------------------------------
def evaluate_rules(rows):
    """
    Extract rule-based predictions from the CSV and score them against
    the manual labels. Returns a metrics dict and the number of skipped rows.
    """
    y_true, y_pred_rules, skipped = convert_to_binary(rows)
    metrics = compute_metrics(y_true, y_pred_rules)
    return metrics, skipped, len(y_true)


# ---------------------------------------------------------------------------
# ML evaluation
# Trains the TF-IDF + Logistic Regression pipeline inside cross-validation
# and collects one prediction per sample from the fold where it was held out.
# This is fairer than training on all data and testing on the same data.
# ---------------------------------------------------------------------------
def evaluate_ml(labeled_file):
    """
    Run stratified cross-validation and return a metrics dict.
    Uses cross_val_predict so every sample gets a fair out-of-sample prediction.
    """
    texts, labels, skipped = load_labeled_data(labeled_file)

    if len(texts) == 0 or len(set(labels)) < 2:
        return None, skipped, 0

    n_folds  = safe_fold_count(labels, max_folds=5)
    pipeline = build_pipeline()
    cv       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # cross_val_predict returns one prediction per sample, produced by the
    # fold in which that sample was the held-out test set.
    # We suppress UndefinedMetricWarning for tiny folds.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred_ml = cross_val_predict(pipeline, texts, labels, cv=cv)

    metrics = compute_metrics(list(labels), list(y_pred_ml))
    return metrics, skipped, len(texts), n_folds


# ---------------------------------------------------------------------------
# Side-by-side table printer
# ---------------------------------------------------------------------------
def print_comparison(rules_m, rules_n, ml_m, ml_n, ml_folds, skipped):
    W = 62   # table width

    def row(label, rules_val, ml_val, highlight=False):
        # Mark the better value with an asterisk
        if isinstance(rules_val, float) and isinstance(ml_val, float):
            r_str = f"{rules_val:.1%}"
            m_str = f"{ml_val:.1%}"
            if rules_val > ml_val:
                r_str = r_str + " *"
            elif ml_val > rules_val:
                m_str = m_str + " *"
        else:
            r_str = str(rules_val)
            m_str = str(ml_val)
        print(f"  {label:<14} {r_str:^18} {m_str:^18}")

    print()
    print("=" * W)
    print("  MODEL COMPARISON — Fake Review Detection")
    print("=" * W)
    print(f"  {'':14} {'Rule-based':^18} {'TF-IDF + LR':^18}")
    print(f"  {'':14} {'(score_reviews)':^18} {'(train_ml_baseline)':^18}")
    print("-" * W)

    row("Labeled rows",   rules_n,           ml_n)
    row("Rows skipped",   skipped,            skipped)
    row("CV folds",       "n/a",              ml_folds)
    print("-" * W)
    row("Accuracy",       rules_m["accuracy"],  ml_m["accuracy"])
    row("Precision",      rules_m["precision"], ml_m["precision"])
    row("Recall",         rules_m["recall"],    ml_m["recall"])
    row("F1 Score",       rules_m["f1"],        ml_m["f1"])
    print("-" * W)
    print(f"  Confusion matrix        Rule-based          TF-IDF + LR")
    print(f"  {'TP (caught)':14} {rules_m['tp']:^18} {ml_m['tp']:^18}")
    print(f"  {'TN (correct)':14} {rules_m['tn']:^18} {ml_m['tn']:^18}")
    print(f"  {'FP (alarm)':14} {rules_m['fp']:^18} {ml_m['fp']:^18}")
    print(f"  {'FN (missed)':14} {rules_m['fn']:^18} {ml_m['fn']:^18}")
    print("=" * W)
    print("  * = better value for that metric")
    print()


# ---------------------------------------------------------------------------
# Automatic interpretation block
# Reads the metrics and writes plain-English thesis observations.
# ---------------------------------------------------------------------------
def print_interpretation(rules_m, ml_m):
    print("-" * 62)
    print("  INTERPRETATION")
    print("-" * 62)

    comparisons = {
        "Accuracy":  ("accuracy",  rules_m["accuracy"],  ml_m["accuracy"]),
        "Precision": ("precision", rules_m["precision"], ml_m["precision"]),
        "Recall":    ("recall",    rules_m["recall"],    ml_m["recall"]),
        "F1 Score":  ("f1",       rules_m["f1"],        ml_m["f1"]),
    }

    for label, (_, r_val, m_val) in comparisons.items():
        diff = abs(r_val - m_val)
        if diff < 0.01:
            winner = "Both models are roughly equal"
        elif r_val > m_val:
            winner = f"Rule-based is higher"
        else:
            winner = f"ML baseline is higher"
        print(f"  {label:<12}: {winner} ({r_val:.1%} vs {m_val:.1%})")

    print()

    # Precision vs recall trade-off commentary
    if rules_m["precision"] > ml_m["precision"] and rules_m["recall"] < ml_m["recall"]:
        print("  Trade-off: The rule-based system is more precise (fewer false")
        print("  alarms) but misses more suspicious reviews (lower recall).")
        print("  The ML model catches more suspicious reviews at the cost of")
        print("  more false alarms.")
    elif ml_m["precision"] > rules_m["precision"] and ml_m["recall"] < rules_m["recall"]:
        print("  Trade-off: The ML model is more precise but has lower recall.")
        print("  The rule-based system catches more suspicious reviews overall.")
    else:
        print("  No clear precision/recall trade-off between the two models.")

    print()
    print("  Note: Dataset is small. These results are a preliminary baseline.")
    print("  Differences of a few percentage points may not be meaningful.")
    print("  Collect more labeled reviews to draw stronger conclusions.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Loading: {LABELED_FILE}\n")

    # Load the full labeled CSV (used for rule-based evaluation)
    rows = load_labeled_csv(LABELED_FILE)
    if not rows:
        print("Could not load labeled data. Check that data/labeled_reviews.csv exists.")
        exit(1)

    print(f"Total rows in file : {len(rows)}")

    # ---- Rule-based ----
    rules_metrics, skipped, rules_n = evaluate_rules(rows)

    if rules_n == 0:
        print("No usable labeled rows. Fill in manual_label in the CSV and re-run.")
        exit(0)

    # ---- ML baseline ----
    ml_result = evaluate_ml(LABELED_FILE)
    ml_metrics, _, ml_n, ml_folds = ml_result

    if ml_metrics is None:
        print("ML evaluation failed. Need at least one sample of each class.")
        exit(1)

    # ---- Print comparison ----
    print_comparison(rules_metrics, rules_n, ml_metrics, ml_n, ml_folds, skipped)
    print_interpretation(rules_metrics, ml_metrics)
