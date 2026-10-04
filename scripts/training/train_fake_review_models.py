# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
train_fake_review_models.py
---------------------------
Trains 8 ML classifiers on data/fake reviews dataset.csv for fake review detection.

Labels: CG (Computer Generated) = 1 (fake)
        OR (Original / Real)    = 0 (genuine)

Each model is saved to data/fake_<name>.joblib.
KNN trains on a 10,000-row balanced sample to keep prediction time manageable.
All other models train on the full 40,432 rows.

HOW TO RUN
----------
    pip install xgboost          # only needed for XGBoost
    python train_fake_review_models.py

OUTPUT
------
    Accuracy / Precision / Recall / F1 printed per model (80/20 test split)
    Models saved to data/fake_*.joblib
"""

import csv
import os
import random
import warnings

import joblib
import numpy as np
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

DATA_FILE  = os.path.join(os.path.dirname(__file__), "data", "fake reviews dataset.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "data")
KNN_SAMPLE = 10_000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(sample_size=None):
    cg_texts, or_texts = [], []
    with open(DATA_FILE, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            label = row.get("label", "").strip().upper()
            text  = row.get("text_", "").strip()
            if label == "CG":
                cg_texts.append(text)
            elif label == "OR":
                or_texts.append(text)

    if sample_size:
        half = sample_size // 2
        random.seed(42)
        cg_texts = random.sample(cg_texts, min(half, len(cg_texts)))
        or_texts = random.sample(or_texts, min(half, len(or_texts)))

    texts  = cg_texts + or_texts
    labels = [1] * len(cg_texts) + [0] * len(or_texts)
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
def print_metrics(name, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"  Accuracy: {acc:.1%}  Precision: {prec:.1%}  Recall: {rec:.1%}  F1: {f1:.1%}")


def train_all():
    print(f"Loading fake review data from: {DATA_FILE}\n")

    all_texts, all_labels = load_data()
    knn_texts, knn_labels = load_data(sample_size=KNN_SAMPLE)

    X_tr,     X_te,     y_tr,     y_te     = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    X_knn_tr, X_knn_te, y_knn_tr, y_knn_te = train_test_split(
        knn_texts, knn_labels, test_size=0.2, random_state=42, stratify=knn_labels)

    W = 52
    print("=" * W)
    print("FAKE REVIEW DETECTION — Training 8 classifiers")
    print("=" * W)

    for name, pipeline, sample_size in get_model_defs():
        print(f"\n[{name.upper()}] Training ...")
        Xtr = X_knn_tr if sample_size else X_tr
        Xte = X_knn_te if sample_size else X_te
        ytr = y_knn_tr if sample_size else y_tr
        yte = y_knn_te if sample_size else y_te

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(Xtr, ytr)
            y_pred = pipeline.predict(Xte)

        print_metrics(name, yte, y_pred)

        # Retrain on full data before saving
        full_texts = knn_texts if sample_size else all_texts
        full_labels = knn_labels if sample_size else all_labels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(full_texts, full_labels)

        path = os.path.join(MODEL_DIR, f"fake_{name}.joblib")
        joblib.dump(pipeline, path)
        print(f"  Saved → {path}")

    print(f"\n{'=' * W}")
    print("Done. All models saved to data/fake_*.joblib")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    train_all()
