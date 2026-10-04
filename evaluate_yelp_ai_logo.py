"""
evaluate_yelp_ai_logo.py
------------------------
Leave-One-Generator-Out (LOGO) evaluation for the AI-detection side of the
generalized multi-modal Yelp detector.

In-distribution AI recall (reported by train_yelp_multimodal_augmented.py) can be
optimistic: the model has seen every generator during training. The honest question
is: does it catch reviews from a generator it has NEVER seen? For each generator G we
RETRAIN the fusion with all of G's AI reviews removed from train+val, then measure
recall on G's reviews. High LOGO recall = the text branch learned a transferable
"AI-written" signal, not generator-specific fingerprints.

Mirrors the methodology of evaluate_cgor_generalization.py (CG/OR LOGO).

RUN
    python evaluate_yelp_ai_logo.py
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "yelp_multimodal_augmented.csv")
DROP_FEATS = {"user_reviews_on_this_biz"}
META_COLS  = {"text", "label", "split", "source", "fake_type", "generator"}
AI_WEIGHT  = 10.0


def log(m):
    print(m, flush=True)


def best_threshold(y_true, proba):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (proba >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def fusion_clf():
    if HAS_XGB:
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                             tree_method="hist", n_jobs=-1, random_state=42)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=42)


def fit_fold(df, exclude_gen, feat_cols):
    """Train text+fusion on all data EXCEPT exclude_gen's AI; return (model, vec, lr, thr)."""
    is_excluded = (df.fake_type == "ai_fake") & (df.generator == exclude_gen)
    tr = df[(df.split == "train") & ~is_excluded]
    va = df[(df.split == "val") & ~is_excluded]

    vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True,
                          min_df=2, stop_words="english", dtype=np.float32)
    Xtr = vec.fit_transform(tr.text.values)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr.fit(Xtr, tr.label.values)

    def feats(sub):
        ts = lr.predict_proba(vec.transform(sub.text.values))[:, 1].reshape(-1, 1)
        return np.hstack([sub[feat_cols].values.astype(np.float32), ts])

    Ftr = feats(tr)
    sw = np.where(tr.fake_type.values == "ai_fake", AI_WEIGHT, 1.0)
    clf = fusion_clf(); clf.fit(Ftr, tr.label.values, sample_weight=sw)
    thr = best_threshold(va.label.values, clf.predict_proba(feats(va))[:, 1])
    return clf, vec, lr, thr, feat_cols


def main():
    df = pd.read_csv(DATA_FILE)
    df["text"] = df["text"].astype(str)
    df["generator"] = df["generator"].fillna("")
    feat_cols = [c for c in df.columns if c not in META_COLS and c not in DROP_FEATS]
    gens = sorted(df[df.fake_type == "ai_fake"]["generator"].unique())
    log(f"generators: {gens}  | features: {len(feat_cols)}")

    log("\n" + "=" * 72)
    log("LEAVE-ONE-GENERATOR-OUT — AI recall on an UNSEEN generator (fusion)")
    log("=" * 72)
    log(f"{'held-out generator':<22}{'n':>7}{'LOGO recall':>14}")
    log("-" * 72)
    recalls = []
    for g in gens:
        clf, vec, lr, thr, fc = fit_fold(df, g, feat_cols)
        held = df[(df.fake_type == "ai_fake") & (df.generator == g)]
        ts = lr.predict_proba(vec.transform(held.text.values))[:, 1].reshape(-1, 1)
        F = np.hstack([held[fc].values.astype(np.float32), ts])
        rec = float((clf.predict_proba(F)[:, 1] >= thr).mean())
        recalls.append(rec)
        log(f"{g:<22}{len(held):>7}{rec:>13.1%}")
    log("-" * 72)
    log(f"mean LOGO AI recall (unseen-generator catch rate): {np.mean(recalls):.1%}")
    log("\nLOGO recall = fraction of an UNSEEN generator's reviews caught as fake.")
    log("Compare to the in-distribution per-generator recall (train script) to read the gap.")


if __name__ == "__main__":
    main()
