# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
evaluate_sentiment_accuracy.py
--------------------------------
Evaluates all 9 sentiment models against Amazon star ratings as ground truth.

Ground truth mapping:
  4-5 stars → positive (1)
  1-2 stars → negative (0)
  3 stars   → skipped (ambiguous)

Samples a balanced set of reviews from Reviews.csv, runs all 9 models,
then prints and saves a comparison table.

HOW TO RUN
----------
    python evaluate_sentiment_accuracy.py

OUTPUT
------
    Terminal: comparison table with accuracy, precision, recall, F1 per model
    data/sentiment_evaluation_results.csv
"""

import csv
import os
import random
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from score_reviews import (
    _run_models, _run_lstm_sentiment, _run_bert_sentiment, _run_distilbert_onnx_sentiment,
    SENTIMENT_MODEL_NAMES, SENTIMENT_CLASS_NAMES,
)

REVIEWS_FILE = os.path.join(os.path.dirname(__file__), "data", "Reviews.csv")
OUTPUT_FILE  = os.path.join(os.path.dirname(__file__), "data", "sentiment_evaluation_results.csv")

SAMPLE_SIZE  = 10_000   # balanced: 5k positive + 5k negative
SEED         = 42

ALL_MODELS   = SENTIMENT_MODEL_NAMES + ["lstm", "bert", "distilbert"]


# ---------------------------------------------------------------------------
# Load balanced sample from Reviews.csv using star ratings as ground truth
# ---------------------------------------------------------------------------
def load_evaluation_data(sample_size=SAMPLE_SIZE):
    """
    Load reviews from Reviews.csv.
    4-5 stars → positive (1), 1-2 stars → negative (0), 3 stars skipped.
    Returns a balanced sample of (text, true_label) tuples.
    """
    pos_reviews, neg_reviews = [], []

    with open(REVIEWS_FILE, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            text = row.get("Text", "").strip()
            if not text:
                continue
            try:
                score = int(float(row.get("Score", "0")))
            except (ValueError, TypeError):
                continue

            if score >= 4:
                pos_reviews.append((text, 1))
            elif score <= 2:
                neg_reviews.append((text, 0))

    half = sample_size // 2
    random.seed(SEED)
    pos_sample = random.sample(pos_reviews, min(half, len(pos_reviews)))
    neg_sample = random.sample(neg_reviews, min(half, len(neg_reviews)))

    combined = pos_sample + neg_sample
    random.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
def print_table(results):
    W = 70
    print()
    print("=" * W)
    print("  SENTIMENT MODEL EVALUATION — Ground truth: Amazon star ratings")
    print("  4-5 stars = positive | 1-2 stars = negative | 3 stars skipped")
    print("=" * W)
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * W)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    for model, m in sorted_results:
        print(f"  {model:<22} {m['accuracy']:>8.1%} {m['precision']:>9.1%} "
              f"{m['recall']:>8.1%} {m['f1']:>7.1%}")
    print("=" * W)
    best = sorted_results[0]
    print(f"\n  Best model by F1: {best[0]}  (F1 = {best[1]['f1']:.1%})")
    print()


# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------
def save_csv(results, sample_size):
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "precision", "recall", "f1", "sample_size"])
        for model, m in sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True):
            writer.writerow([
                model,
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                sample_size,
            ])
    print(f"  Results saved to: {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading {SAMPLE_SIZE:,} balanced reviews from Reviews.csv ...")
    data       = load_evaluation_data(SAMPLE_SIZE)
    texts      = [t for t, _ in data]
    true_labels = [l for _, l in data]
    actual_size = len(data)
    print(f"Loaded {actual_size:,} reviews  "
          f"({sum(true_labels):,} positive / {actual_size - sum(true_labels):,} negative)\n")

    results = {}

    # --- sklearn models ---
    print("Running sklearn sentiment models ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = _run_models(SENTIMENT_MODEL_NAMES, "sentiment", texts, SENTIMENT_CLASS_NAMES)

    for name in SENTIMENT_MODEL_NAMES:
        y_pred = [p[name]["label"] for p in preds]
        if any(l is None for l in y_pred):
            print(f"  [{name}] skipped — model not trained yet")
            continue
        results[name] = metrics(true_labels, y_pred)
        print(f"  [{name}] F1 = {results[name]['f1']:.1%}")

    # --- LSTM ---
    print("\nRunning LSTM model ...")
    lstm_preds = _run_lstm_sentiment(texts)
    y_pred_lstm = [p["label"] for p in lstm_preds]
    if any(l is None for l in y_pred_lstm):
        print("  [lstm] skipped — model not available")
    else:
        results["lstm"] = metrics(true_labels, y_pred_lstm)
        print(f"  [lstm] F1 = {results['lstm']['f1']:.1%}")

    # --- BERT --- (commented out — slow on CPU, I ran this file with BERT once and obtained F1=88.2%)
    # print("\nRunning BERT model ...")
    # bert_preds = _run_bert_sentiment(texts)
    # y_pred_bert = [p["label"] for p in bert_preds]
    # if any(l is None for l in y_pred_bert):
    #     print("  [bert] skipped — model not available")
    # else:
    #     results["bert"] = metrics(true_labels, y_pred_bert)
    #     print(f"  [bert] F1 = {results['bert']['f1']:.1%}")

    # --- DistilBERT ONNX ---
    print("\nRunning DistilBERT ONNX model ...")
    distilbert_preds = _run_distilbert_onnx_sentiment(texts)
    y_pred_distilbert = [p["label"] for p in distilbert_preds]
    if any(l is None for l in y_pred_distilbert):
        print("  [distilbert] skipped — model not available")
    else:
        results["distilbert"] = metrics(true_labels, y_pred_distilbert)
        print(f"  [distilbert] F1 = {results['distilbert']['f1']:.1%}")

    if not results:
        print("\nNo models available. Train the models first.")
        return

    print_table(results)
    save_csv(results, actual_size)


if __name__ == "__main__":
    main()
