"""
train_combined_fake_sklearn.py — 8 sklearn UNIFIED fake detectors that flag a review fake
if it is EITHER computer-generated (CG) OR human-deceptive.

Trained on data/combined_fake_reviews.csv (CG/OR + Ott/Li human deception). The human rows
are ~2.5% of fakes, so they're OVERSAMPLED in TRAIN (x HUMAN_OVERSAMPLE) so the model learns
both fake types (val/test untouched -> real distribution). Stop words are KEPT (a deception
cue). The headline metric is PER-FAKE-TYPE recall on test: does it catch CG fakes AND human
fakes while sparing genuine reviews?

Each model -> data/combined_fake_models/<name>.joblib + thresholds.json
RUN:  python train_combined_fake_sklearn.py
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
from sklearn.metrics import f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "combined_fake_reviews.csv")
OUT  = os.path.join(HERE, "data", "combined_fake_models")
HUMAN_OVERSAMPLE = 15
KNN_SAMPLE = 20_000


def pipe(clf, maxf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=maxf, ngram_range=(1, 2), sublinear_tf=True,
                                  min_df=2)),                      # stop words KEPT (deception cue)
        ("clf", clf),
    ])


def models():
    m = [
        ("knn", KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"), 5000, True),
        ("rf",  RandomForestClassifier(n_estimators=200, max_depth=40, min_samples_leaf=5,
                                       class_weight="balanced", random_state=42, n_jobs=-1), 5000, False),
        ("dt",  DecisionTreeClassifier(class_weight="balanced", random_state=42), 5000, False),
        ("lr",  LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42), 20000, False),
        ("nb",  MultinomialNB(), 20000, False),
        ("adaboost", AdaBoostClassifier(n_estimators=100, random_state=42), 5000, False),
        ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=40, early_stopping=True, random_state=42), 20000, False),
    ]
    if HAS_XGB:
        m.insert(5, ("xgb", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                          eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=42), 5000, False))
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
    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)

    # oversample human-origin TRAIN rows so the unified model learns human deception too
    hum = tr[tr.origin == "human"]
    tr_os = pd.concat([tr] + [hum] * (HUMAN_OVERSAMPLE - 1), ignore_index=True)
    print(f"train {len(tr):,} -> {len(tr_os):,} after {HUMAN_OVERSAMPLE}x human oversample "
          f"(human train rows {len(hum):,}); val {len(va):,} / test {len(te):,}")
    ft_te = te["fake_type"].values

    def per_type(pred):
        # recall on each fake type + genuine specificity
        cg = (pred[ft_te == "cg_fake"] == 1).mean() if (ft_te == "cg_fake").any() else float("nan")
        hu = (pred[ft_te == "human_fake"] == 1).mean() if (ft_te == "human_fake").any() else float("nan")
        ge = (pred[ft_te == "genuine"] == 0).mean() if (ft_te == "genuine").any() else float("nan")
        return cg, hu, ge

    rng = np.random.RandomState(42)
    knn_idx = rng.choice(len(tr_os), size=min(KNN_SAMPLE, len(tr_os)), replace=False)

    thresholds, rows, te_pred = {}, [], {}
    for name, clf, maxf, is_knn in models():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = pipe(clf, maxf)
            if is_knn:
                p.fit(tr_os.text.iloc[knn_idx], tr_os.label.values[knn_idx])
            else:
                p.fit(tr_os.text, tr_os.label.values)
            thr = best_thr(va.label.values, p.predict_proba(va.text)[:, 1])
            proba = p.predict_proba(te.text)[:, 1]
        pred = (proba >= thr).astype(int)
        te_pred[name] = pred
        cg, hu, ge = per_type(pred)
        rows.append((name, thr, f1_score(te.label, pred, average="macro", zero_division=0), cg, hu, ge))
        thresholds[name] = thr
        joblib.dump(p, os.path.join(OUT, f"{name}.joblib"))
    json.dump(thresholds, open(os.path.join(OUT, "thresholds.json"), "w"), indent=2)

    M = np.vstack([te_pred[n] for n, *_ in rows])
    ens = (M.sum(0) > M.shape[0] / 2).astype(int)
    ecg, ehu, ege = per_type(ens)
    ens_row = ("ENSEMBLE", float("nan"), f1_score(te.label, ens, average="macro", zero_division=0), ecg, ehu, ege)

    print("\n" + "=" * 74)
    print("UNIFIED FAKE DETECTOR (CG + human) — test  | recall per fake type + genuine spec")
    print("=" * 74)
    print(f"{'model':<11}{'thr':>6}{'macroF1':>9}{'CG-fakeR':>10}{'humanR':>9}{'genuineSpec':>13}")
    for name, thr, mf1, cg, hu, ge in sorted(rows, key=lambda r: -r[2]) + [ens_row]:
        t = f"{thr:.2f}" if thr == thr else "  -"
        print(f"{name:<11}{t:>6}{mf1:>8.1%}{cg:>10.1%}{hu:>9.1%}{ge:>12.1%}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
