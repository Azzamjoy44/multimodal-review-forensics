# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
train_ml_baseline.py
---------------------
Trains a simple supervised ML model on the manually labeled reviews and
compares it to the rule-based baseline from evaluate_against_labels.py.

Pipeline:
  TF-IDF (bag-of-words counts) → Logistic Regression

Because the dataset is small, we evaluate using stratified cross-validation
instead of a fixed train/test split, which gives a more stable estimate.

HOW TO RUN
----------
    pip install scikit-learn
    python train_ml_baseline.py

OUTPUT
------
  - Evaluation metrics printed to the terminal
  - Trained model saved to: data/ml_review_model.joblib
"""

import csv
import os

import warnings

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

# Reuse the same metric function used by evaluate_against_labels.py and
# compare_models.py so all three scripts report identical numbers.
from evaluate_against_labels import compute_metrics

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
LABELED_FILE     = os.path.join(os.path.dirname(__file__), "data", "labeled_reviews.csv")
KAGGLE_FAKE_FILE = os.path.join(os.path.dirname(__file__), "data", "fake reviews dataset.csv")
MODEL_FILE       = os.path.join(os.path.dirname(__file__), "data", "ml_review_model.joblib")

# Labels accepted as genuine (0) or suspicious (1)
LABEL_TO_BINARY = {
    "genuine":    0,
    "suspicious": 1,
    "fake":       1,   # synonym for suspicious
}

# Kaggle fake-reviews dataset labels
# CG = Computer Generated (fake/suspicious), OR = Original (genuine)
KAGGLE_LABEL_TO_BINARY = {
    "CG": 1,
    "OR": 0,
}

# Rows with these manual_label values are skipped
SKIP_LABELS = {"", "uncertain"}


# ---------------------------------------------------------------------------
# Step 1 — Load the labeled CSV and return texts + binary labels
# ---------------------------------------------------------------------------
def load_labeled_data(path):
    """
    Read data/labeled_reviews.csv.
    Returns:
      texts  — list of review_text strings
      labels — list of int  (0 = genuine, 1 = suspicious)
      skipped — number of rows ignored (blank / uncertain)
    """
    if not os.path.exists(path):
        print(f"Error: '{path}' not found.")
        print("Make sure data/labeled_reviews.csv exists and is filled in.")
        return [], [], 0

    texts   = []
    labels  = []
    skipped = 0

    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manual = row.get("manual_label", "").strip().lower()

            # Skip blank / uncertain rows
            if manual in SKIP_LABELS:
                skipped += 1
                continue

            # Skip unrecognised labels with a warning
            if manual not in LABEL_TO_BINARY:
                print(f"Warning: Unknown label '{manual}' on {row.get('review_id')} — skipped.")
                skipped += 1
                continue

            texts.append(row.get("review_text", ""))
            labels.append(LABEL_TO_BINARY[manual])

    return texts, labels, skipped


# ---------------------------------------------------------------------------
# Step 1b — Load Kaggle fake-reviews dataset
# ---------------------------------------------------------------------------
def load_kaggle_fake_data(path=None, sample_size=None):
    """
    Read data/fake reviews dataset.csv.

    Labels in the file:
      CG  (Computer Generated) → 1  (suspicious / fake)
      OR  (Original / Real)    → 0  (genuine)

    sample_size: if given, return a balanced random sample of this many rows
                 (half CG, half OR).  Pass None to use all rows.

    Returns:
      texts   — list of review text strings
      labels  — list of int (0 = genuine, 1 = suspicious)
      skipped — number of rows skipped (unknown labels)
    """
    import random

    if path is None:
        path = KAGGLE_FAKE_FILE

    if not os.path.exists(path):
        print(f"Error: '{path}' not found.")
        return [], [], 0

    cg_texts = []
    or_texts = []
    skipped  = 0

    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "").strip().upper()
            text  = row.get("text_", "").strip()
            if label == "CG":
                cg_texts.append(text)
            elif label == "OR":
                or_texts.append(text)
            else:
                skipped += 1

    if sample_size is not None:
        half = sample_size // 2
        random.seed(42)
        cg_texts = random.sample(cg_texts, min(half, len(cg_texts)))
        or_texts = random.sample(or_texts, min(half, len(or_texts)))

    texts  = cg_texts + or_texts
    labels = [1] * len(cg_texts) + [0] * len(or_texts)

    # Shuffle so classes are interleaved
    combined = list(zip(texts, labels))
    random.seed(42)
    random.shuffle(combined)
    texts, labels = zip(*combined) if combined else ([], [])

    return list(texts), list(labels), skipped


# ---------------------------------------------------------------------------
# Step 2 — Choose a safe number of CV folds given the minority class size
#
# With very few positive examples, 5-fold CV can produce folds with only
# 1 suspicious review in the test set, making metrics unreliable.
# We cap folds at the size of the smallest class to avoid this.
# ---------------------------------------------------------------------------
def safe_fold_count(labels, max_folds=5):
    """
    Return the largest number of CV folds that is still safe to use.
    'Safe' means every fold will have at least one example of each class.
    Minimum returned value is 2.
    """
    min_class_count = min(labels.count(0), labels.count(1))
    # Allow at most min_class_count folds (so each fold has >= 1 minority sample)
    safe = min(max_folds, min_class_count)
    return max(safe, 2)   # at least 2 folds


# ---------------------------------------------------------------------------
# Step 3 — Build the ML pipeline
#
# TfidfVectorizer converts each review_text into a vector of word importance
# scores (TF-IDF = term frequency × inverse document frequency).
#
# LogisticRegression is a simple linear classifier that learns which words
# push the score towards "suspicious" vs "genuine".
#
# class_weight='balanced' tells the model to give more importance to the
# minority class (suspicious), which helps with the class imbalance.
# ---------------------------------------------------------------------------
def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # use single words AND two-word phrases
            min_df=1,             # include all tokens (dataset is small)
            sublinear_tf=True,    # dampen very frequent words
        )),
        ("lr", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,        # enough iterations for small datasets
            random_state=42,
        )),
    ])


# ---------------------------------------------------------------------------
# Step 4 — Print a clear summary of the cross-validation results
#
# METRIC DEFINITION (same as evaluate_against_labels.py and compare_models.py)
# ---------------------------------------------------------------------------
# All metrics are computed on the SUSPICIOUS class (positive class = 1):
#
#   Precision = TP / (TP + FP)
#     Of all reviews the model flagged as suspicious, how many truly were?
#
#   Recall    = TP / (TP + FN)
#     Of all truly suspicious reviews, how many did the model catch?
#
#   F1        = harmonic mean of Precision and Recall
#
# Predictions come from cross_val_predict (stratified k-fold), so every
# sample is predicted by a model that was NOT trained on that sample.
# This gives fair out-of-sample estimates even without a held-out test set.
# ---------------------------------------------------------------------------
def print_report(labels, skipped, metrics, n_folds):
    n_genuine    = labels.count(0)
    n_suspicious = labels.count(1)
    total        = len(labels)

    print()
    print("=" * 58)
    print("ML BASELINE — TF-IDF + Logistic Regression")
    print("=" * 58)
    print(f"  Labeled rows used  : {total}")
    print(f"  Rows skipped       : {skipped}  (blank or 'uncertain')")
    print(f"  Class distribution : {n_genuine} genuine  /  {n_suspicious} suspicious")
    print(f"  Evaluation method  : {n_folds}-fold stratified cross_val_predict")
    print(f"  Metric focus       : suspicious class (positive = 1)")
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
    print("=" * 58)
    print()
    print("  [!] SMALL DATASET WARNING")
    print(f"     Only {total} labeled examples ({n_suspicious} suspicious).")
    print("     These numbers are a rough preliminary baseline only.")
    print("     Metrics will be much more reliable with 200+ labeled rows.")
    print()


# ---------------------------------------------------------------------------
# Step 5 — Print the most informative words found by the model
#
# Logistic Regression assigns a coefficient to each TF-IDF feature (word).
# Positive coefficient → the word pushes the review towards 'suspicious'.
# Negative coefficient → the word pushes the review towards 'genuine'.
# ---------------------------------------------------------------------------
def print_top_features(pipeline, n=10):
    feature_names = pipeline["tfidf"].get_feature_names_out()
    coefs         = pipeline["lr"].coef_[0]   # one row for binary classification

    # Sort by coefficient value
    ranked = sorted(zip(coefs, feature_names))

    print(f"  Top {n} words/phrases associated with SUSPICIOUS reviews:")
    for coef, word in reversed(ranked[-n:]):
        print(f"    {coef:+.3f}  '{word}'")

    print()
    print(f"  Top {n} words/phrases associated with GENUINE reviews:")
    for coef, word in ranked[:n]:
        print(f"    {coef:+.3f}  '{word}'")
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def train_and_evaluate(source="kaggle"):
    """
    Train and evaluate the ML pipeline.

    source:
      "manual"   — use data/labeled_reviews.csv only (small manual labels)
      "kaggle"   — use data/fake reviews dataset.csv (40k labeled reviews)
      "combined" — merge both sources
    """
    if source == "kaggle":
        print(f"Loading Kaggle labeled data from: {KAGGLE_FAKE_FILE}\n")
        texts, labels, skipped = load_kaggle_fake_data()
        data_desc = "Kaggle fake-reviews dataset (CG / OR)"
    elif source == "combined":
        print("Loading combined data (manual + Kaggle) ...\n")
        t1, l1, s1 = load_labeled_data(LABELED_FILE)
        t2, l2, s2 = load_kaggle_fake_data()
        texts, labels, skipped = t1 + t2, l1 + l2, s1 + s2
        data_desc = f"Combined: {len(t1)} manual + {len(t2)} Kaggle rows"
    else:
        print(f"Loading labeled data from: {LABELED_FILE}\n")
        texts, labels, skipped = load_labeled_data(LABELED_FILE)
        data_desc = "Manual labels (labeled_reviews.csv)"

    print(f"Data source: {data_desc}")

    if len(texts) == 0:
        print("No usable labeled rows found. Fill in manual_label in the CSV first.")
        return

    # Check that both classes are present
    if len(set(labels)) < 2:
        print("Error: Need at least one 'genuine' AND one 'suspicious' label to train.")
        print(f"  Found only: {set(labels)}")
        return

    # Choose a safe number of folds
    n_folds = safe_fold_count(labels, max_folds=5)
    if n_folds < 5:
        print(f"Note: Using {n_folds}-fold CV (reduced from 5) because the minority "
              f"class has only {min(labels.count(0), labels.count(1))} examples.\n")

    # Build the pipeline and run cross-validation.
    # cross_val_predict returns one prediction per sample, produced by the
    # fold where that sample was held out (never seen during training).
    # This is the same method used in compare_models.py so results are identical.
    pipeline = build_pipeline()
    cv       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # silence warnings from tiny folds
        y_pred = cross_val_predict(pipeline, texts, labels, cv=cv)

    # compute_metrics uses binary precision/recall for the suspicious class,
    # identical to the computation in evaluate_against_labels.py and compare_models.py
    metrics = compute_metrics(list(labels), list(y_pred))

    # Print evaluation summary
    print_report(labels, skipped, metrics, n_folds)

    # Retrain on the full labeled dataset so the saved model uses all available data
    print("Retraining on all labeled data to save the final model...")
    pipeline.fit(texts, labels)

    # Show the most informative words
    print_top_features(pipeline, n=10)

    # Save the trained pipeline to disk
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")
    print("  You can load it later with:  model = joblib.load('data/ml_review_model.joblib')")
    print("  And predict with:            model.predict(['review text here'])")


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "kaggle"
    if src not in ("manual", "kaggle", "combined"):
        print(f"Unknown source '{src}'. Use: manual | kaggle | combined")
        sys.exit(1)
    train_and_evaluate(source=src)
