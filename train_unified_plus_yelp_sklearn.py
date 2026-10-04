"""
train_unified_plus_yelp_sklearn.py — ABLATION: does folding Yelp into the unified text
detector help? (Spoiler: no — Yelp's label is behavioral, not textual.)

Trains the SAME text-only sklearn setup TWO ways and compares per-fake-type recall:
  * A = "unified"        : CG/OR + Ott/Li human deception  (the current unified objective)
  * B = "unified + Yelp" : the same, PLUS a balanced sample of YelpZip+NYC+Chi review TEXT
                           (label = Yelp's behavioral spam flag)

Both are evaluated on the SAME held-out tests, broken out by fake type:
  - CG-fake recall + human-fake recall + genuine specificity  (combined test)
  - Yelp-fake recall + Yelp-genuine specificity + Yelp macro-F1  (yelp_split test)

EXPECTED RESULT (the thesis point): adding Yelp text does NOT lift Yelp detection above the
~chance/69% text ceiling (the signal isn't in the text), and it DEGRADES CG + human recall
(normal-looking restaurant text labeled "fake" injects label noise). Confirms the modality
argument: Yelp needs the behavioral (Model B) / relational (GNN) modality, not more text.

Representative fast+strong subset of the 8 unified models (lr/nb/mlp/xgb) + their majority,
for a quick local run. Stop words KEPT (matches the unified trainer). No GPU, ~minutes.
RUN:  python train_unified_plus_yelp_sklearn.py
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

HERE = os.path.dirname(__file__)
COMBINED = os.path.join(HERE, "data", "combined_fake_reviews.csv")
YELP     = os.path.join(HERE, "data", "yelp_multimodal_features_full.csv")
HUMAN_OVERSAMPLE = 15
YELP_TRAIN_PER_CLASS = 30_000   # balanced Yelp peer class (don't let 642k Yelp dominate)


def pipe(clf, maxf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=maxf, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("clf", clf),
    ])


def models():
    m = [
        ("lr",  LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42), 20000),
        ("nb",  MultinomialNB(), 20000),
        ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=40, early_stopping=True, random_state=42), 20000),
    ]
    if HAS_XGB:
        m.append(("xgb", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                       eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=42), 5000))
    return m


def best_thr(y, proba):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y, (proba >= t).astype(int), average="macro", zero_division=0)
        if f > bf:
            bf, bt = f, t
    return float(bt)


def load_data():
    # combined (CG/OR + human)
    cg = pd.read_csv(COMBINED)[["text", "label", "fake_type", "split", "origin"]]
    cg["text"] = cg["text"].astype(str)

    # Yelp full (text + behavioral label only — we feed TEXT to a text model on purpose)
    yp = pd.read_csv(YELP, usecols=["text", "label", "split"])
    yp["text"] = yp["text"].astype(str)
    yp["fake_type"] = np.where(yp["label"] == 1, "yelp_fake", "yelp_genuine")
    yp["origin"] = "yelp"

    # balanced Yelp TRAIN peer sample (val/test kept whole for eval)
    yt = yp[yp.split == "train"]
    rng = np.random.RandomState(42)
    f_idx = yt.index[yt.label == 1]; g_idx = yt.index[yt.label == 0]
    n = min(YELP_TRAIN_PER_CLASS, len(f_idx), len(g_idx))
    keep = np.concatenate([rng.choice(f_idx, n, replace=False), rng.choice(g_idx, n, replace=False)])
    yelp_train = yp.loc[keep]
    yelp_val   = yp[yp.split == "val"]
    yelp_test  = yp[yp.split == "test"]
    print(f"combined: {len(cg):,} | yelp train sample {len(yelp_train):,} (bal {n:,}/class) "
          f"| yelp val {len(yelp_val):,} | yelp test {len(yelp_test):,}", flush=True)
    return cg, yelp_train, yelp_val, yelp_test


def oversample_human(tr):
    hum = tr[tr.origin == "human"]
    return pd.concat([tr] + [hum] * (HUMAN_OVERSAMPLE - 1), ignore_index=True)


def evaluate(te_pred_c, te_c_ft, te_pred_y, te_y_label):
    """per-type recall / specificity from the two test sets."""
    def rec(pred, mask, want):  # recall (want=1) or specificity (want=0)
        if not mask.any(): return float("nan")
        return (pred[mask] == want).mean()
    cg = rec(te_pred_c, te_c_ft == "cg_fake", 1)
    hu = rec(te_pred_c, te_c_ft == "human_fake", 1)
    ge = rec(te_pred_c, te_c_ft == "genuine", 0)
    yf = rec(te_pred_y, te_y_label == 1, 1)
    yg = rec(te_pred_y, te_y_label == 0, 0)
    ymf1 = f1_score(te_y_label, te_pred_y, average="macro", zero_division=0)
    return cg, hu, ge, yf, yg, ymf1


def run_variant(name, train_df, val_df, te_c, te_y):
    """fit the model set on train_df, threshold on val_df, predict both test sets; majority."""
    tr = oversample_human(train_df)
    ft_c = te_c["fake_type"].values
    yl   = te_y["label"].values
    preds_c, preds_y, per = {}, {}, []
    for mname, clf, maxf in models():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = pipe(clf, maxf)
            p.fit(tr.text, tr.label.values)
            thr = best_thr(val_df.label.values, p.predict_proba(val_df.text)[:, 1])
            pc = (p.predict_proba(te_c.text)[:, 1] >= thr).astype(int)
            py = (p.predict_proba(te_y.text)[:, 1] >= thr).astype(int)
        preds_c[mname] = pc; preds_y[mname] = py
        per.append((mname, *evaluate(pc, ft_c, py, yl)))
        print(f"  [{name}] {mname} done", flush=True)
    Mc = np.vstack(list(preds_c.values())); My = np.vstack(list(preds_y.values()))
    ec = (Mc.sum(0) > Mc.shape[0] / 2).astype(int); ey = (My.sum(0) > My.shape[0] / 2).astype(int)
    per.append(("MAJORITY", *evaluate(ec, ft_c, ey, yl)))
    return per


def main():
    cg, yelp_train, yelp_val, yelp_test = load_data()
    te_c = cg[cg.split == "test"].reset_index(drop=True)
    va_c = cg[cg.split == "val"].reset_index(drop=True)
    tr_c = cg[cg.split == "train"].reset_index(drop=True)

    print("\n=== A: unified (CG + human) — no Yelp ===", flush=True)
    A = run_variant("A", tr_c, va_c, te_c, yelp_test)

    print("\n=== B: unified + Yelp text ===", flush=True)
    trB = pd.concat([tr_c, yelp_train], ignore_index=True)
    vaB = pd.concat([va_c, yelp_val], ignore_index=True)
    B = run_variant("B", trB, vaB, te_c, yelp_test)

    hdr = f"{'model':<10}{'CG-R':>7}{'humR':>7}{'genSpec':>9} | {'yelpFR':>7}{'yelpGSpec':>10}{'yelpF1':>8}"
    def show(title, rows):
        print("\n" + "=" * 60); print(title); print("=" * 60); print(hdr)
        for nm, cg_, hu, ge, yf, yg, ym in rows:
            print(f"{nm:<10}{cg_:>6.1%}{hu:>7.1%}{ge:>8.1%} | {yf:>7.1%}{yg:>9.1%}{ym:>8.1%}")
    show("A — UNIFIED (no Yelp)", A)
    show("B — UNIFIED + YELP TEXT", B)

    # headline delta on the majority row
    da = dict((r[0], r) for r in A)["MAJORITY"]; db = dict((r[0], r) for r in B)["MAJORITY"]
    print("\n" + "=" * 60)
    print("DELTA (B - A), majority — did adding Yelp help?")
    print("=" * 60)
    print(f"  CG-fake recall : {da[1]:.1%} -> {db[1]:.1%}  ({(db[1]-da[1])*100:+.1f} pp)")
    print(f"  human recall   : {da[2]:.1%} -> {db[2]:.1%}  ({(db[2]-da[2])*100:+.1f} pp)")
    print(f"  genuine spec   : {da[3]:.1%} -> {db[3]:.1%}  ({(db[3]-da[3])*100:+.1f} pp)")
    print(f"  YELP fake recall: {da[4]:.1%} -> {db[4]:.1%}  ({(db[4]-da[4])*100:+.1f} pp)")
    print(f"  YELP macro-F1   : {da[6]:.1%} -> {db[6]:.1%}  ({(db[6]-da[6])*100:+.1f} pp)  (text ceiling ~69%)")
    print("\nIf Yelp recall stays ~chance while CG/human drop -> modality mismatch confirmed.")


if __name__ == "__main__":
    main()
