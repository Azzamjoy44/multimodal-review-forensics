# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
train_yelp_fake_models.py
--------------------------
Trains 8 sklearn classifiers on the combined YelpZip + YelpChi datasets
for fake review detection.

Labels:
  YelpZip  — metadata column 4:  -1 = fake (1),  1 = genuine (0)
  YelpChi  — metadata column 5:   Y = fake (1),  N = genuine (0)

Output: data/yelp_fake_<name>.joblib

HOW TO RUN
----------
    python train_yelp_fake_models.py
"""

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

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
KNN_SAMPLE = 20_000   # KNN is slow on large datasets; train on a balanced sample


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_subdir(prefix):
    """Return the first data/ subdirectory whose name starts with prefix."""
    for name in os.listdir(DATA_DIR):
        if name.startswith(prefix):
            return os.path.join(DATA_DIR, name, prefix)
    return None


def load_yelpzip():
    """Load YelpZip reviews and labels from metadata + reviewContent files."""
    base = _find_subdir("YelpZip")
    if not base:
        print("  [WARNING] YelpZip directory not found — skipping.")
        return [], []

    metadata_path = os.path.join(base, "metadata")
    review_path   = os.path.join(base, "reviewContent")

    if not os.path.exists(metadata_path) or not os.path.exists(review_path):
        print("  [WARNING] YelpZip metadata or reviewContent missing — skipping.")
        return [], []

    texts, labels = [], []
    with open(metadata_path, encoding="utf-8", errors="ignore") as mf, \
         open(review_path,   encoding="utf-8", errors="ignore") as rf:
        for meta_line, review_line in zip(mf, rf):
            m = meta_line.strip().split("\t")
            r = review_line.strip().split("\t")
            if len(m) < 4 or len(r) < 4:
                continue
            label_str = m[3]
            if label_str == "-1":
                label = 1   # fake
            elif label_str == "1":
                label = 0   # genuine
            else:
                continue
            text = r[3].strip()
            if text:
                texts.append(text)
                labels.append(label)

    return texts, labels


def load_yelpchi():
    """Load YelpChi restaurant + hotel reviews from metadata + review text files."""
    base = _find_subdir("YelpChi")
    if not base:
        print("  [WARNING] YelpChi directory not found — skipping.")
        return [], []

    texts, labels = [], []
    for subset in ("yelpResData", "yelpHotelData"):
        meta_path   = os.path.join(base, f"output_meta_{subset}_NRYRcleaned.txt")
        review_path = os.path.join(base, f"output_review_{subset}_NRYRcleaned.txt")
        if not os.path.exists(meta_path) or not os.path.exists(review_path):
            print(f"  [WARNING] YelpChi {subset} files missing — skipping subset.")
            continue
        with open(meta_path,   encoding="utf-8", errors="ignore") as mf, \
             open(review_path, encoding="utf-8", errors="ignore") as rf:
            for meta_line, review_line in zip(mf, rf):
                parts = meta_line.strip().split()
                if len(parts) < 5:
                    continue
                label_str = parts[4]
                if label_str == "Y":
                    label = 1   # fake
                elif label_str == "N":
                    label = 0   # genuine
                else:
                    continue
                text = review_line.strip()
                if text:
                    texts.append(text)
                    labels.append(label)

    return texts, labels


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def make_pipeline(clf, max_features=20_000):
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

def get_model_defs(n_genuine, n_fake):
    """Build model list with class weights tuned for the Yelp imbalance ratio."""
    ratio = n_genuine / max(n_fake, 1)   # ~6.6 for YelpZip
    defs = [
        ("knn", make_pipeline(
            KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"),
            max_features=5_000,
        ), KNN_SAMPLE),
        ("rf", make_pipeline(
            RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                   random_state=42, n_jobs=-1),
            max_features=10_000,
        ), None),
        ("dt", make_pipeline(
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
            max_features=10_000,
        ), None),
        ("lr", make_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            max_features=20_000,
        ), None),
        ("nb", make_pipeline(
            MultinomialNB(),
            max_features=20_000,
        ), None),
        ("adaboost", make_pipeline(
            AdaBoostClassifier(n_estimators=50, random_state=42),
            max_features=5_000,
        ), None),
        ("mlp", make_pipeline(
            MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42),
            max_features=20_000,
        ), None),
    ]
    if HAS_XGB:
        defs.insert(5, ("xgb", make_pipeline(
            XGBClassifier(
                n_estimators=200,
                scale_pos_weight=ratio,   # compensate for genuine/fake imbalance
                random_state=42,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
            ),
            max_features=10_000,
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


def make_balanced_sample(texts, labels, n, seed=42):
    """Return a balanced sample of n/2 fake + n/2 genuine rows."""
    fake_idx    = [i for i, l in enumerate(labels) if l == 1]
    genuine_idx = [i for i, l in enumerate(labels) if l == 0]
    rng = random.Random(seed)
    half = n // 2
    fake_idx    = rng.sample(fake_idx,    min(half, len(fake_idx)))
    genuine_idx = rng.sample(genuine_idx, min(half, len(genuine_idx)))
    idx = fake_idx + genuine_idx
    rng.shuffle(idx)
    return [texts[i] for i in idx], [labels[i] for i in idx]


def train_all():
    print("Loading YelpZip ...")
    zip_texts, zip_labels = load_yelpzip()
    print(f"  {len(zip_texts):,} reviews loaded")

    print("Loading YelpChi ...")
    chi_texts, chi_labels = load_yelpchi()
    print(f"  {len(chi_texts):,} reviews loaded")

    all_texts  = zip_texts + chi_texts
    all_labels = zip_labels + chi_labels

    if not all_texts:
        print("No data found. Make sure the YelpZip and YelpChi folders are in data/.")
        return

    n_fake    = sum(all_labels)
    n_genuine = len(all_labels) - n_fake
    print(f"\nCombined: {len(all_texts):,} reviews")
    print(f"  Fake   : {n_fake:,}  ({n_fake/len(all_labels):.1%})")
    print(f"  Genuine: {n_genuine:,}  ({n_genuine/len(all_labels):.1%})")

    # Shuffle combined data
    combined = list(zip(all_texts, all_labels))
    random.seed(42)
    random.shuffle(combined)
    all_texts, all_labels = zip(*combined)
    all_texts  = list(all_texts)
    all_labels = list(all_labels)

    X_tr, X_te, y_tr, y_te = train_test_split(
        all_texts, all_labels, test_size=0.1, random_state=42, stratify=all_labels
    )
    knn_texts, knn_labels = make_balanced_sample(all_texts, all_labels, KNN_SAMPLE)
    X_knn_tr, X_knn_te, y_knn_tr, y_knn_te = train_test_split(
        knn_texts, knn_labels, test_size=0.1, random_state=42, stratify=knn_labels
    )

    W = 56
    print("\n" + "=" * W)
    print("YELP FAKE DETECTION — Training 8 classifiers")
    print("=" * W)

    for name, pipeline, sample_size in get_model_defs(n_genuine, n_fake):
        print(f"\n[{name.upper()}] Training ...")
        Xtr = X_knn_tr if sample_size else X_tr
        Xte = X_knn_te if sample_size else X_te
        ytr = y_knn_tr if sample_size else y_tr
        yte = y_knn_te if sample_size else y_te

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(Xtr, ytr)
            y_pred = pipeline.predict(Xte)

        print_metrics(yte, y_pred)

        # Retrain on full split (no held-out) before saving
        full_texts  = knn_texts if sample_size else all_texts
        full_labels = knn_labels if sample_size else all_labels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(full_texts, full_labels)

        path = os.path.join(DATA_DIR, f"yelp_fake_{name}.joblib")
        joblib.dump(pipeline, path)
        print(f"  Saved → {path}")

    print(f"\n{'=' * W}")
    print("Done. All Yelp fake models saved to data/yelp_fake_*.joblib")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    train_all()
