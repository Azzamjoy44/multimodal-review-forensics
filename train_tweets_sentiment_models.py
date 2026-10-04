"""
train_tweets_sentiment_models.py
---------------------------------
Trains 8 ML sentiment classifiers on data/tweets.csv (Sentiment140 dataset)
and saves each model to data/sentiment_<name>.joblib, replacing any previously
trained sentiment models so the frontend uses these automatically.

Dataset format (no header row):
  col 0 — label  : 0 = negative, 4 = positive
  col 1 — tweet ID
  col 2 — date
  col 3 — query
  col 4 — user
  col 5 — tweet text

Because the dataset has 1.6 million rows, training uses a balanced random
sample (default 100,000 rows). Change SAMPLE_SIZE below to use more or fewer.

HOW TO RUN
----------
    pip install xgboost          # only needed for XGBoost
    python train_tweets_sentiment_models.py

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

DATA_FILE   = os.path.join(os.path.dirname(__file__), "data", "tweets.csv")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "data")
SAMPLE_SIZE = 100_000   # balanced: 50k negative + 50k positive
KNN_SAMPLE  = 10_000    # KNN subset: 5k negative + 5k positive


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(sample_size=SAMPLE_SIZE):
    """
    Load tweets.csv and return a balanced sample.
    Labels: 0 = negative, 4 = positive → mapped to 0 and 1.
    """
    neg_texts, pos_texts = [], []

    with open(DATA_FILE, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            label = row[0].strip()
            text  = row[5].strip()
            if label == "0":
                neg_texts.append(text)
            elif label == "4":
                pos_texts.append(text)

    half = sample_size // 2
    random.seed(42)
    neg_sample = random.sample(neg_texts, min(half, len(neg_texts)))
    pos_sample = random.sample(pos_texts, min(half, len(pos_texts)))

    texts  = neg_sample + pos_sample
    labels = [0] * len(neg_sample) + [1] * len(pos_sample)

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
            RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                   random_state=42, n_jobs=-1),
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
    print(f"Loading tweets data from: {DATA_FILE}")
    print(f"Using balanced sample of {SAMPLE_SIZE:,} tweets ({SAMPLE_SIZE//2:,} per class)\n")

    all_texts, all_labels = load_data(SAMPLE_SIZE)
    knn_texts, knn_labels = load_data(KNN_SAMPLE)

    X_tr,     X_te,     y_tr,     y_te     = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    X_knn_tr, X_knn_te, y_knn_tr, y_knn_te = train_test_split(
        knn_texts, knn_labels, test_size=0.2, random_state=42, stratify=knn_labels)

    W = 54
    print("=" * W)
    print("TWEET SENTIMENT — Training 8 classifiers")
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
