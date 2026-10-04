# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
train_fake_broadened_sklearn.py
-------------------------------
Trains 8 sklearn classifiers for BROADENED fake-review detection (CG vs OR =
machine-generated vs human) on data/fake_reviews_broadened.csv
(built by build_broadened_cgor.py).

Labels:  1 = CG (machine-generated / fake),  0 = OR (original / human).

Uses the dataset's pre-made split column (train / val / test): fit on train,
tune each model's threshold for best macro-F1 on val, report on test. The TF-IDF
is fit once per feature-size and reused (float32) for speed/memory.

Like the original Amazon fake models (and unlike the review-detector), this KEEPS
stop_words="english" — consistent with the project's fake-detection pipeline.

Each model  -> data/fake_broadened_sklearn_models/<name>.joblib
Thresholds  -> data/fake_broadened_sklearn_models/thresholds.json

HOW TO RUN
----------
    python train_fake_broadened_sklearn.py
"""

import os
import gc
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)
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

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "fake_reviews_broadened.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "data", "fake_broadened_sklearn_models")
KNN_SAMPLE = 20_000


def make_vectorizer(max_features):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        stop_words="english",   # matches the original fake-detection pipeline
        dtype=np.float32,
    )


def get_models():
    """(name, classifier, tfidf_max_features, is_knn)."""
    models = [
        ("knn", KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"), 5_000, True),
        ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                      random_state=42, n_jobs=-1), 5_000, False),
        ("dt", DecisionTreeClassifier(class_weight="balanced", random_state=42), 5_000, False),
        ("adaboost", AdaBoostClassifier(n_estimators=50, random_state=42), 5_000, False),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42), 10_000, False),
        ("nb", MultinomialNB(), 10_000, False),
        ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=40,
                              early_stopping=True, n_iter_no_change=5, random_state=42), 10_000, False),
    ]
    if HAS_XGB:
        models.append(("xgb", XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss",
                                            tree_method="hist", n_jobs=-1), 5_000, False))
    else:
        print("XGBoost not installed - skipping. Run: pip install xgboost")
    return models


def best_threshold(y_true, proba):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (proba >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def report(name, y_true, pred, thr):
    print(f"  thr={thr:.2f}  acc={accuracy_score(y_true, pred):.3%}  "
          f"fakeP={precision_score(y_true, pred, zero_division=0):.3%}  "
          f"fakeR={recall_score(y_true, pred, zero_division=0):.3%}  "
          f"fakeF1={f1_score(y_true, pred, zero_division=0):.3%}  "
          f"macroF1={f1_score(y_true, pred, average='macro', zero_division=0):.3%}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    df["text"] = df["text"].astype(str)

    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    y_tr, y_va, y_te = tr["label"].values, va["label"].values, te["label"].values
    print(f"train={len(tr)}  val={len(va)}  test={len(te)}  (1=CG/fake, 0=OR/human)")

    rng = np.random.RandomState(42)
    knn_pos = rng.choice(len(tr), size=min(KNN_SAMPLE, len(tr)), replace=False)

    models = get_models()
    thresholds = {}
    print("=" * 60)
    print("BROADENED FAKE REVIEW DETECTION (CG vs OR) - sklearn")
    print("=" * 60)

    for max_features in sorted({m[2] for m in models}):
        group = [m for m in models if m[2] == max_features]
        print(f"\n--- TF-IDF max_features={max_features:,} (reused by {[m[0] for m in group]}) ---")
        vec = make_vectorizer(max_features)
        Xtr = vec.fit_transform(tr["text"].values)
        Xva = vec.transform(va["text"].values)
        Xte = vec.transform(te["text"].values)

        for name, clf, _, is_knn in group:
            print(f"\n[{name.upper()}] training ...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if is_knn:
                    clf.fit(Xtr[knn_pos], y_tr[knn_pos])
                else:
                    clf.fit(Xtr, y_tr)
                thr = best_threshold(y_va, clf.predict_proba(Xva)[:, 1])
                te_proba = clf.predict_proba(Xte)[:, 1]
            report(name, y_te, (te_proba >= thr).astype(int), thr)
            thresholds[name] = thr
            joblib.dump(Pipeline([("tfidf", vec), ("clf", clf)]),
                        os.path.join(MODEL_DIR, f"{name}.joblib"))
            print(f"  saved -> {name}.joblib")

        del Xtr, Xva, Xte, vec
        gc.collect()

    with open(os.path.join(MODEL_DIR, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nwrote thresholds.json: {thresholds}")
    print("Done.")


if __name__ == "__main__":
    main()
