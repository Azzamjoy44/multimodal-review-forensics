"""
train_merged_sentiment_models.py
---------------------------------
Trains 8 ML sentiment classifiers on data/merged_sentiment.csv
(the combined IMDB + tweets dataset) and saves each model to
data/sentiment_<name>.joblib, replacing any previously trained
sentiment models so the frontend picks them up automatically.

Merged dataset: ~1,650,000 rows, label 0 = negative, 1 = positive.
Training uses a balanced random sample (default 500,000 rows) drawn
equally from both classes.  Change SAMPLE_SIZE below to use more or fewer.

HOW TO RUN
----------
    python merge_sentiment_datasets.py   # create merged CSV first
    python train_merged_sentiment_models.py

OUTPUT
------
    Accuracy / Precision / Recall / F1 per model (80/20 test split)
    Models saved to data/sentiment_*.joblib
"""

import csv
import os
import random
import warnings

import joblib
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

MERGED_FILE = os.path.join(os.path.dirname(__file__), "data", "merged_sentiment.csv")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "data")
SAMPLE_SIZE = 500_000   # balanced: 250k negative + 250k positive
KNN_SAMPLE  = 10_000    # KNN subset: 5k negative + 5k positive


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(sample_size=SAMPLE_SIZE):
    """
    Load a balanced sample from merged_sentiment.csv.

    All IMDB rows are always included. The remaining slots are filled
    with a random sample of tweets so both sources are guaranteed
    to be represented.

    merged_sentiment.csv layout: IMDB rows first, then tweets.
    IMDB total: 50,000 (25k positive + 25k negative).
    """
    if not os.path.exists(MERGED_FILE):
        raise FileNotFoundError(
            f"'{MERGED_FILE}' not found. "
            "Run merge_sentiment_datasets.py first."
        )

    imdb_pos, imdb_neg = [], []
    tweet_pos, tweet_neg = [], []
    imdb_done = False
    imdb_count = 0

    with open(MERGED_FILE, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "").strip()
            text  = row.get("text",  "").strip()
            if not text:
                continue

            if not imdb_done:
                # IMDB rows come first in the merged file (50,000 total)
                if label == "1":
                    imdb_pos.append(text)
                elif label == "0":
                    imdb_neg.append(text)
                imdb_count += 1
                if imdb_count >= 50_000:
                    imdb_done = True
            else:
                if label == "1":
                    tweet_pos.append(text)
                elif label == "0":
                    tweet_neg.append(text)

    half = sample_size // 2
    random.seed(42)

    if half >= len(imdb_pos):
        # Sample large enough to include all IMDB rows — fill the rest with tweets
        tweet_slots      = half - len(imdb_pos)
        tweet_neg_slots  = half - len(imdb_neg)
        tweet_pos_sample = random.sample(tweet_pos, min(tweet_slots,     len(tweet_pos)))
        tweet_neg_sample = random.sample(tweet_neg, min(tweet_neg_slots, len(tweet_neg)))
    else:
        # Sample too small to fit all IMDB rows — draw randomly from the full pool
        imdb_pos, imdb_neg = [], []
        tweet_pos_sample   = random.sample(tweet_pos, min(half, len(tweet_pos)))
        tweet_neg_sample   = random.sample(tweet_neg, min(half, len(tweet_neg)))

    pos_texts = imdb_pos + tweet_pos_sample
    neg_texts = imdb_neg + tweet_neg_sample

    texts  = pos_texts + neg_texts
    labels = [1] * len(pos_texts) + [0] * len(neg_texts)

    combined = list(zip(texts, labels))
    random.seed(42)
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------
def make_pipeline(clf, max_features=10_000):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            stop_words="english",
        )),
        ("clf", clf),
    ])


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def get_model_defs():
    defs = [
        ("knn", make_pipeline(
            KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"),
            max_features=5_000,
        ), KNN_SAMPLE),
        ("rf", make_pipeline(
            # max_depth + min_samples_leaf bound the tree size: without them the 100
            # trees grow until pure on 500k samples -> a ~2 GB model. Bounding shrinks it
            # ~10-40x with negligible accuracy change (deep unbounded RF on TF-IDF overfits).
            RandomForestClassifier(n_estimators=100, max_depth=40, min_samples_leaf=5,
                                   class_weight="balanced", random_state=42, n_jobs=-1),
            max_features=5_000,
        ), None),
        ("dt", make_pipeline(
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
            max_features=5_000,
        ), None),
        ("lr", make_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            max_features=10_000,
        ), None),
        ("nb", make_pipeline(
            MultinomialNB(),
            max_features=10_000,
        ), None),
        ("adaboost", make_pipeline(
            AdaBoostClassifier(n_estimators=50, random_state=42),
            max_features=5_000,
        ), None),
        ("mlp", make_pipeline(
            MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
            max_features=10_000,
        ), None),
    ]
    if HAS_XGB:
        defs.insert(5, ("xgb", make_pipeline(
            XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss",
                          tree_method="hist", n_jobs=-1),
            max_features=5_000,
        ), None))
    else:
        print("XGBoost not installed — skipping. Run: pip install xgboost")
    return defs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def print_metrics(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"  Accuracy: {acc:.1%}  Precision: {prec:.1%}  Recall: {rec:.1%}  F1: {f1:.1%}")


def train_all():
    print(f"Loading merged sentiment data from: {MERGED_FILE}")
    print(f"Sample: {SAMPLE_SIZE:,} rows — all 50,000 IMDB reviews + {SAMPLE_SIZE-50_000:,} tweets\n")

    all_texts, all_labels = load_data(SAMPLE_SIZE)
    knn_texts, knn_labels = load_data(KNN_SAMPLE)

    X_tr,     X_te,     y_tr,     y_te     = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    X_knn_tr, X_knn_te, y_knn_tr, y_knn_te = train_test_split(
        knn_texts, knn_labels, test_size=0.2, random_state=42, stratify=knn_labels)

    W = 56
    print("=" * W)
    print("MERGED SENTIMENT (IMDB + Tweets) — Training 8 classifiers")
    print("=" * W)

    for name, pipeline, use_knn_sample in get_model_defs():
        print(f"\n[{name.upper()}] Training ...")
        Xtr = X_knn_tr if use_knn_sample else X_tr
        Xte = X_knn_te if use_knn_sample else X_te
        ytr = y_knn_tr if use_knn_sample else y_tr
        yte = y_knn_te if use_knn_sample else y_te

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(Xtr, ytr)
            y_pred = pipeline.predict(Xte)

        print_metrics(yte, y_pred)

        full_texts  = knn_texts if use_knn_sample else all_texts
        full_labels = knn_labels if use_knn_sample else all_labels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(full_texts, full_labels)

        path = os.path.join(MODEL_DIR, f"sentiment_{name}.joblib")
        joblib.dump(pipeline, path)
        print(f"  Saved → {path}")

    print(f"\n{'=' * W}")
    print("Done. All sentiment models saved to data/sentiment_*.joblib")
    print("Restart the server to pick up the new models.")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    train_all()
