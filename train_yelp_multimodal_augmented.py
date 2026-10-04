"""
train_yelp_multimodal_augmented.py
----------------------------------
Phase 2 training: the GENERALIZED multi-modal Yelp detector, trained on the
AI-augmented dataset (data/yelp_multimodal_augmented.csv from
build_yelp_multimodal_augmented.py).

Same recipe as train_yelp_multimodal_fusion.py (text TF-IDF+LR with out-of-fold
train scores + behavioural XGBoost, fused with XGBoost), but the test report adds
a per-fake-type breakdown so we can see each modality's job:

    * human_fake recall  — Yelp behavioural spam (behavioural branch's job)
    * ai_fake recall      — AI-generated reviews   (text branch's job)
    * genuine specificity — false-positive control

Expectation: behavioural-only catches human fakes but misses AI fakes; text-only
catches AI fakes but is weak on human fakes; FUSION catches BOTH -> generalized.

RUN
    python train_yelp_multimodal_augmented.py
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "yelp_multimodal_augmented.csv")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "data", "yelp_multimodal_generalized_models")
DROP_FEATS = {"user_reviews_on_this_biz"}
META_COLS  = {"text", "label", "split", "source", "fake_type", "generator"}
AI_WEIGHT  = 10.0   # upweight AI fakes in the fusion so the behavioural branch can't drown
                    # out the text signal (raises AI recall without moving the threshold)


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
    log("XGBoost not installed — using HistGradientBoosting.")
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=42)


def report(name, y_te, proba, thr, ftype, rows):
    pred = (proba >= thr).astype(int)
    macro = f1_score(y_te, pred, average="macro", zero_division=0)
    acc = accuracy_score(y_te, pred)
    # per-type
    def rec(mask):
        m = mask.values
        return float(pred[m].mean()) if m.sum() else float("nan")  # fraction flagged fake
    human_r = rec(ftype == "human_fake")
    ai_r    = rec(ftype == "ai_fake")
    gen_spec = 1.0 - rec(ftype == "genuine")   # genuine correctly kept
    rows.append({"model": name, "thr": round(thr, 2), "acc": acc, "macro_f1": macro,
                 "human_fake_recall": human_r, "ai_fake_recall": ai_r,
                 "genuine_specificity": gen_spec})


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    df["text"] = df["text"].astype(str)
    feat_cols = [c for c in df.columns if c not in META_COLS and c not in DROP_FEATS]

    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    y_tr, y_va, y_te = tr.label.values, va.label.values, te.label.values
    ftype_te = te["fake_type"]
    log(f"train={len(tr):,} val={len(va):,} test={len(te):,} | features={len(feat_cols)}")
    log("test fake_type counts: " + te["fake_type"].value_counts().to_dict().__str__())

    # ---- TEXT modality: leakage-free OOF train scores -------------------- #
    log("\nfitting text model (TF-IDF + LR) ...")
    vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True,
                          min_df=2, stop_words="english", dtype=np.float32)
    Xtr = vec.fit_transform(tr.text.values)
    Xva = vec.transform(va.text.values)
    Xte = vec.transform(te.text.values)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts_tr = cross_val_predict(lr, Xtr, y_tr, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
        lr.fit(Xtr, y_tr)
    ts_va = lr.predict_proba(Xva)[:, 1]
    ts_te = lr.predict_proba(Xte)[:, 1]

    Btr = tr[feat_cols].values.astype(np.float32)
    Bva = va[feat_cols].values.astype(np.float32)
    Bte = te[feat_cols].values.astype(np.float32)

    rows = []
    # text-only
    report("text_only (TF-IDF+LR)", y_te, ts_te, best_threshold(y_va, ts_va), ftype_te, rows)
    # behavioural-only
    log("training behavioural-only ...")
    cb = fusion_clf(); cb.fit(Btr, y_tr)
    report("behavioural_only", y_te, cb.predict_proba(Bte)[:, 1],
           best_threshold(y_va, cb.predict_proba(Bva)[:, 1]), ftype_te, rows)
    # fusion
    log("training FUSION (text + behavioural) ...")
    Ftr = np.hstack([Btr, ts_tr.reshape(-1, 1)])
    Fva = np.hstack([Bva, ts_va.reshape(-1, 1)])
    Fte = np.hstack([Bte, ts_te.reshape(-1, 1)])
    sw = np.where(tr["fake_type"].values == "ai_fake", AI_WEIGHT, 1.0)
    cf = fusion_clf(); cf.fit(Ftr, y_tr, sample_weight=sw)
    pf_te = cf.predict_proba(Fte)[:, 1]
    thr_f = best_threshold(y_va, cf.predict_proba(Fva)[:, 1])
    report(f"FUSION (AI weight {AI_WEIGHT:g})", y_te, pf_te, thr_f, ftype_te, rows)

    log("\n" + "=" * 92)
    log("GENERALIZED MULTI-MODAL — test results (Yelp human fakes + injected AI fakes)")
    log("=" * 92)
    log(f"{'model':<26}{'thr':>5}{'acc':>8}{'macroF1':>9}"
        f"{'humanFakeR':>12}{'aiFakeR':>10}{'genuineSpec':>13}")
    log("-" * 92)
    for r in rows:
        log(f"{r['model']:<26}{r['thr']:>5.2f}{r['acc']:>7.1%} {r['macro_f1']:>8.1%}"
            f"{r['human_fake_recall']:>11.1%}{r['ai_fake_recall']:>10.1%}"
            f"{r['genuine_specificity']:>13.1%}")
    log("-" * 92)
    log("humanFakeR = Yelp behavioural spam caught | aiFakeR = AI reviews caught | genuineSpec = genuine kept")

    # per-generator AI recall (fusion): how well it catches each generator family
    gen_te = te["generator"].values
    pred_f = (pf_te >= thr_f).astype(int)
    ai_mask = ftype_te == "ai_fake"
    log("\nFUSION per-generator AI-fake recall (in-distribution):")
    for g in sorted(set(gen_te[ai_mask])):
        m = ai_mask & (gen_te == g)
        log(f"  {g:<10}{pred_f[m].mean():>6.1%}  (n={int(m.sum())})")

    import joblib
    joblib.dump(vec, os.path.join(OUT_DIR, "text_tfidf.joblib"))
    joblib.dump(lr,  os.path.join(OUT_DIR, "text_lr.joblib"))
    joblib.dump(cf,  os.path.join(OUT_DIR, "fusion_clf.joblib"))
    json.dump({"feat_cols": feat_cols, "threshold": thr_f},
              open(os.path.join(OUT_DIR, "fusion_meta.json"), "w"), indent=2)
    log(f"\nsaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
