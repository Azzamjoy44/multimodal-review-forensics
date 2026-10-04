# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
train_human_fake_sklearn.py — 8 sklearn detectors of HUMAN-WRITTEN deceptive reviews,
trained on the Ott + Li 2014 gold corpus (data/human_fake/human_fake_reviews.csv).

This is a distinct capability from the project's other fake detectors: Model A catches
AI/computer-generated reviews; the Yelp models use behavioural labels; these catch
*human deception* (people paid to write fake reviews) from the text alone.

Design note — UNLIKE the CG/OR and sentiment models, this KEEPS English stop words.
Deception research (Ott et al. 2011) shows function words / personal-pronoun usage are
strong deception cues, so we do NOT strip them.

Each model -> data/human_fake/human_fake_sklearn_models/<name>.joblib + thresholds.json
RUN:  python train_human_fake_sklearn.py
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "human_fake", "human_fake_reviews.csv")
OUT  = os.path.join(HERE, "data", "human_fake", "human_fake_sklearn_models")


def pipe(clf, maxf=20000):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=maxf, ngram_range=(1, 2), sublinear_tf=True,
                                  min_df=2)),                     # NOTE: stop words KEPT (deception cue)
        ("clf", clf),
    ])


def models():
    m = [
        ("knn", KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute")),
        ("rf",  RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)),
        ("dt",  DecisionTreeClassifier(class_weight="balanced", random_state=42)),
        ("lr",  LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ("nb",  MultinomialNB()),
        ("adaboost", AdaBoostClassifier(n_estimators=100, random_state=42)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, early_stopping=True, random_state=42)),
    ]
    if HAS_XGB:
        m.insert(5, ("xgb", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                                          eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=42)))
    return m


def best_thr(y, proba):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y, (proba >= t).astype(int), average="macro", zero_division=0)
        if f > bf:
            bf, bt = f, t
    return float(bt)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA); df["text"] = df["text"].astype(str)
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    print(f"train {len(tr):,} / val {len(va):,} / test {len(te):,}  "
          f"(test: {int((te.label==1).sum())} fake / {int((te.label==0).sum())} genuine)\n")

    thresholds, rows, te_preds = {}, [], {}
    for name, clf in models():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = pipe(clf); p.fit(tr.text, tr.label)
            thr = best_thr(va.label.values, p.predict_proba(va.text)[:, 1])
            proba = p.predict_proba(te.text)[:, 1]
        pred = (proba >= thr).astype(int)
        te_preds[name] = pred
        rows.append((name, thr, f1_score(te.label, pred, average="macro", zero_division=0),
                     recall_score(te.label, pred, pos_label=1, zero_division=0),
                     precision_score(te.label, pred, pos_label=1, zero_division=0)))
        thresholds[name] = thr
        joblib.dump(p, os.path.join(OUT, f"{name}.joblib"))
    json.dump(thresholds, open(os.path.join(OUT, "thresholds.json"), "w"), indent=2)

    # majority-vote ensemble
    M = np.vstack([te_preds[n] for n, *_ in rows])
    ens = (M.sum(0) > M.shape[0] / 2).astype(int)
    ens_row = ("ENSEMBLE", float("nan"), f1_score(te.label, ens, average="macro", zero_division=0),
               recall_score(te.label, ens, pos_label=1, zero_division=0),
               precision_score(te.label, ens, pos_label=1, zero_division=0))

    print("=" * 64)
    print("HUMAN-DECEPTION (Ott + Li) — test results")
    print("=" * 64)
    print(f"{'model':<11}{'thr':>6}{'macroF1':>10}{'fakeR':>8}{'fakeP':>8}")
    for name, thr, mf1, fr, fp in sorted(rows, key=lambda r: -r[2]) + [ens_row]:
        t = f"{thr:.2f}" if thr == thr else "  -"
        print(f"{name:<11}{t:>6}{mf1:>9.1%}{fr:>8.1%}{fp:>8.1%}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
