"""
train_yelp_multimodal_sklearn.py
--------------------------------
The 8 sklearn members of the GENERALIZED multi-modal Yelp ensemble (Model B).

Mirrors the project's other sklearn fake detectors (knn/rf/dt/lr/nb/xgb/adaboost/mlp),
but every model is **multi-modal**: it sees the review's **TF-IDF text features**
*concatenated with* the **17 behavioural features** (via a ColumnTransformer), so each
classifier independently fuses both modalities and casts its own fake/genuine vote —
just like the per-model dot-strip in the Amazon/Yelp sections, now multi-modal.

Trained on the AI-augmented dataset (`data/yelp_multimodal_augmented.csv`) so the
ensemble is generalized: the TF-IDF features let each model catch AI-written reviews,
the behavioural features catch human spam.

Design details:
  * Behavioural block is **MinMax-scaled to [0,1]** so it's non-negative (MultinomialNB
    requires that) and on a comparable scale to the sublinear TF-IDF.
  * AI fakes are **up-weighted ×10** (`sample_weight`) for the classifiers that support
    it (rf/dt/lr/nb/xgb/adaboost) so the behavioural signal can't drown out the text
    signal for behaviourally-innocent AI reviews. (knn/mlp don't take sample_weight.)
  * Thresholds tuned on val for macro-F1; reported on test with a per-fake-type breakdown.

Each model  -> data/yelp_multimodal_sklearn_models/<name>.joblib  (a full Pipeline:
              ColumnTransformer[tfidf(text)+minmax(behavioural)] -> classifier)
Thresholds  -> data/yelp_multimodal_sklearn_models/thresholds.json

Serving: pass a DataFrame with a 'text' column + the behavioural feature columns.

RUN
    python train_yelp_multimodal_sklearn.py
"""

import os
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
from sklearn.metrics import f1_score, accuracy_score, recall_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "yelp_multimodal_augmented.csv")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "data", "yelp_multimodal_sklearn_models")
DROP   = {"user_reviews_on_this_biz"}
META   = {"text", "label", "split", "source", "fake_type", "generator", "length_bucket"}
AI_WEIGHT  = 10.0
KNN_SAMPLE = 20_000


def log(m):
    print(m, flush=True)


def get_models():
    """(name, classifier, tfidf_max_features, is_knn)."""
    models = [
        ("knn", KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"), 5_000, True),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=40, min_samples_leaf=5,
                                      class_weight="balanced", random_state=42, n_jobs=-1), 5_000, False),
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
        log("XGBoost not installed - skipping.")
    return models


def make_pipeline(clf, max_features, feat_cols):
    ct = ColumnTransformer([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                  sublinear_tf=True, min_df=2, stop_words="english"), "text"),
        ("beh", MinMaxScaler(), feat_cols),
    ])
    return Pipeline([("feat", ct), ("clf", clf)])


def best_threshold(y_true, proba):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (proba >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    df["text"] = df["text"].astype(str)
    feat_cols = [c for c in df.columns if c not in META and c not in DROP]
    log(f"rows={len(df):,}  behavioural features={len(feat_cols)}")

    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    y_tr, y_va, y_te = tr.label.values, va.label.values, te.label.values
    ftype_te = te["fake_type"].values
    cols = ["text"] + feat_cols
    Xtr, Xva, Xte = tr[cols], va[cols], te[cols]
    sw = np.where(tr["fake_type"].values == "ai_fake", AI_WEIGHT, 1.0)
    log(f"train={len(tr):,} val={len(va):,} test={len(te):,}  "
        f"(test fake_type {pd.Series(ftype_te).value_counts().to_dict()})")

    rng = np.random.RandomState(42)
    knn_idx = rng.choice(len(tr), size=min(KNN_SAMPLE, len(tr)), replace=False)

    thresholds, rows = {}, []
    for name, clf, max_features, is_knn in get_models():
        log(f"\n[{name.upper()}] training (multi-modal: TF-IDF{max_features} + behavioural) ...")
        pipe = make_pipeline(clf, max_features, feat_cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if is_knn:
                pipe.fit(Xtr.iloc[knn_idx], y_tr[knn_idx])          # subsample (brute KNN)
            else:
                try:
                    pipe.fit(Xtr, y_tr, clf__sample_weight=sw)       # AI up-weighting
                except TypeError:
                    pipe.fit(Xtr, y_tr)                              # clf doesn't take sample_weight
            thr = best_threshold(y_va, pipe.predict_proba(Xva)[:, 1])
            proba_te = pipe.predict_proba(Xte)[:, 1]
        pred = (proba_te >= thr).astype(int)

        def rec(mask):
            return float(pred[mask].mean()) if mask.sum() else float("nan")
        rows.append({
            "model": name, "thr": round(thr, 2),
            "macro_f1": f1_score(y_te, pred, average="macro", zero_division=0),
            "human_fake_recall": rec(ftype_te == "human_fake"),
            "ai_fake_recall": rec(ftype_te == "ai_fake"),
            "genuine_spec": 1.0 - rec(ftype_te == "genuine"),
        })
        thresholds[name] = thr
        joblib.dump(pipe, os.path.join(OUT_DIR, f"{name}.joblib"))
        log(f"  saved {name}.joblib  (thr={thr:.2f})")

    with open(os.path.join(OUT_DIR, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    log("\n" + "=" * 80)
    log("MULTI-MODAL SKLEARN ENSEMBLE (Model B) — test results")
    log("=" * 80)
    log(f"{'model':<10}{'thr':>6}{'macroF1':>10}{'humanFakeR':>13}{'aiFakeR':>10}{'genuineSpec':>13}")
    log("-" * 80)
    for r in rows:
        log(f"{r['model']:<10}{r['thr']:>6.2f}{r['macro_f1']:>9.1%} "
            f"{r['human_fake_recall']:>12.1%}{r['ai_fake_recall']:>10.1%}{r['genuine_spec']:>13.1%}")
    log("-" * 80)
    log(f"saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
