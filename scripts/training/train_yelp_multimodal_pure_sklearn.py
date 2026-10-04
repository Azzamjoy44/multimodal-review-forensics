# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
train_yelp_multimodal_pure_sklearn.py — Model B retrained WITHOUT AI reviews.

The served Model B (`train_yelp_multimodal_sklearn.py`) was trained on the AI-augmented
corpus (`yelp_multimodal_augmented.csv`) with AI fakes up-weighted x10 — i.e. optimized for
a *generalized* (AI + human) objective, which miscalibrated it for the pure Yelp human-fake
task it's actually served on. This retrains the same multi-modal ensemble (8 sklearn,
ColumnTransformer: TF-IDF text + MinMax behavioral) on the **pure Yelp** data
(`yelp_multimodal_features.csv`, 178,905 rows, 50/50 human-fake/genuine — no AI, no x10).

Output: data/yelp_multimodal_pure_sklearn_models/ (a NEW dir — does not overwrite the served
models, so we can compare and decide what to serve). Reports test macro-F1 vs the references.
RUN:  python train_yelp_multimodal_pure_sklearn.py
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, precision_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import model_b   # FEAT_COLS (17 behavioral)

HERE = os.path.dirname(__file__)
# optional argv: [data_csv] [out_dir] — defaults to the 178k balanced set
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "yelp_multimodal_features.csv")
OUT  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "data", "yelp_multimodal_pure_sklearn_models")
FEAT = model_b.FEAT_COLS
KNN_SAMPLE = 20_000


def pipe(clf, maxf):
    ct = ColumnTransformer([
        ("tfidf", TfidfVectorizer(max_features=maxf, ngram_range=(1, 2), sublinear_tf=True,
                                  min_df=2, stop_words="english"), "text"),
        ("beh", MinMaxScaler(), FEAT),
    ])
    return Pipeline([("feat", ct), ("clf", clf)])


def models():
    m = [
        ("knn", KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"), 5000, True),
        ("rf",  RandomForestClassifier(n_estimators=100, max_depth=40, min_samples_leaf=5,
                                       class_weight="balanced", random_state=42, n_jobs=-1), 5000, False),
        ("dt",  DecisionTreeClassifier(class_weight="balanced", random_state=42), 5000, False),
        ("lr",  LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42), 10000, False),
        ("nb",  MultinomialNB(), 10000, False),
        ("adaboost", AdaBoostClassifier(n_estimators=50, random_state=42), 5000, False),
        ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=40, early_stopping=True, random_state=42), 10000, False),
    ]
    if HAS_XGB:
        m.insert(5, ("xgb", XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss",
                                          tree_method="hist", n_jobs=-1), 5000, False))
    return m


def best_thr(y, p):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
        if f > bf:
            bf, bt = f, t
    return float(bt)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA); df["text"] = df["text"].astype(str)
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    cols = ["text"] + FEAT
    print(f"PURE Yelp (no AI): train {len(tr):,} / val {len(va):,} / test {len(te):,}", flush=True)
    rng = np.random.RandomState(42)
    knn_idx = rng.choice(len(tr), size=min(KNN_SAMPLE, len(tr)), replace=False)

    thr, rows, te_pred = {}, [], {}
    for name, clf, maxf, is_knn in models():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = pipe(clf, maxf)
            p.fit(tr[cols].iloc[knn_idx] if is_knn else tr[cols],
                  tr.label.values[knn_idx] if is_knn else tr.label.values)
            t = best_thr(va.label.values, p.predict_proba(va[cols])[:, 1])
            proba = p.predict_proba(te[cols])[:, 1]
        pred = (proba >= t).astype(int); te_pred[name] = pred
        rows.append((name, t, f1_score(te.label, pred, average="macro", zero_division=0),
                     recall_score(te.label, pred, pos_label=1, zero_division=0),
                     precision_score(te.label, pred, pos_label=1, zero_division=0)))
        thr[name] = t; joblib.dump(p, os.path.join(OUT, f"{name}.joblib"))
        print(f"  {name} done (macroF1 {rows[-1][2]:.1%})", flush=True)
    json.dump(thr, open(os.path.join(OUT, "thresholds.json"), "w"), indent=2)

    M = np.vstack([te_pred[n] for n, *_ in rows])
    ens = (M.sum(0) > M.shape[0] / 2).astype(int)
    ens_row = ("ENSEMBLE", float("nan"), f1_score(te.label, ens, average="macro", zero_division=0),
               recall_score(te.label, ens, pos_label=1, zero_division=0),
               precision_score(te.label, ens, pos_label=1, zero_division=0))

    print("\n" + "=" * 60)
    print("PURE Model B (text+behavioral, NO AI) — Yelp clean test")
    print("=" * 60)
    print(f"{'model':<11}{'thr':>6}{'macroF1':>10}{'fakeR':>8}{'fakeP':>8}")
    for name, t, mf1, fr, fp in sorted(rows, key=lambda r: -r[2]) + [ens_row]:
        ts = f"{t:.2f}" if t == t else "  -"
        print(f"{name:<11}{ts:>6}{mf1:>9.1%}{fr:>8.1%}{fp:>8.1%}")
    print("\nReferences (same Yelp test): text-only ~69% · behavioral-only ~76% · served AI-aug Model B (confounded).")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
