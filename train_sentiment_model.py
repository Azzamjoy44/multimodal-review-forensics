"""
train_sentiment_model.py
------------------------
Trains a sentiment classifier on the IMDB movie review dataset and saves
the model to data/sentiment_model.joblib.

Pipeline: TF-IDF (unigrams + bigrams) → Logistic Regression

The model learns to distinguish positive from negative reviews and is then
applied to Amazon product reviews on the frontend to show predicted sentiment.

HOW TO RUN
----------
    python train_sentiment_model.py

OUTPUT
------
    Evaluation metrics printed to the terminal
    Trained model saved to: data/sentiment_model.joblib
"""

import csv
import os
import warnings

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from evaluate_against_labels import compute_metrics

IMDB_FILE   = os.path.join(os.path.dirname(__file__), "data", "IMDB Dataset.csv")
MODEL_FILE  = os.path.join(os.path.dirname(__file__), "data", "sentiment_model.joblib")

LABEL_MAP = {"positive": 1, "negative": 0}


def load_imdb_data(path=None):
    """
    Load IMDB Dataset.csv.
    Returns texts (list of str), labels (list of int: 1=positive, 0=negative).
    """
    if path is None:
        path = IMDB_FILE

    if not os.path.exists(path):
        print(f"Error: '{path}' not found.")
        return [], []

    texts  = []
    labels = []

    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentiment = row.get("sentiment", "").strip().lower()
            text      = row.get("review", "").strip()
            if sentiment in LABEL_MAP:
                texts.append(text)
                labels.append(LABEL_MAP[sentiment])

    return texts, labels


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=100_000,
        )),
        ("lr", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])


def train_and_evaluate():
    print(f"Loading IMDB data from: {IMDB_FILE}\n")
    texts, labels = load_imdb_data()

    if not texts:
        print("No data loaded. Check that data/IMDB Dataset.csv exists.")
        return

    n_pos = labels.count(1)
    n_neg = labels.count(0)
    print(f"  Total reviews : {len(texts)}")
    print(f"  Positive      : {n_pos}")
    print(f"  Negative      : {n_neg}\n")

    print("Running 5-fold cross-validation ...")
    pipeline = build_pipeline()
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = cross_val_predict(pipeline, texts, labels, cv=cv)

    metrics = compute_metrics(list(labels), list(y_pred))

    print()
    print("=" * 50)
    print("SENTIMENT MODEL — TF-IDF + Logistic Regression")
    print("=" * 50)
    print(f"  Accuracy  : {metrics['accuracy']:.1%}")
    print(f"  Precision : {metrics['precision']:.1%}")
    print(f"  Recall    : {metrics['recall']:.1%}")
    print(f"  F1 Score  : {metrics['f1']:.1%}")
    print("=" * 50)
    print()

    print("Retraining on full dataset to save final model ...")
    pipeline.fit(texts, labels)

    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    train_and_evaluate()
