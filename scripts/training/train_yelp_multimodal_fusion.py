# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
train_yelp_multimodal_fusion.py
-------------------------------
MULTI-MODAL Yelp fake-review detector — Model #1 (feature-level fusion, local).

Combines TWO modalities into one model / one verdict:
  * TEXT modality      -> a text score P(fake | text) from TF-IDF + LogisticRegression
  * BEHAVIORAL modality-> the engineered reviewer/business features
                          (data/yelp_multimodal_features.csv, built by
                           build_yelp_multimodal_features.py)

To keep the comparison honest, this trains and reports THREE models on the
IDENTICAL train/val/test split (the same split as the text-only Yelp baselines):
    1. text-only        (text score alone)            -> reproduces the ~66-69% ceiling
    2. behavioral-only  (metadata features alone)
    3. FUSION           (text score + behavioral)      -> the multi-modal model

Leakage control: the text score on TRAIN rows is produced out-of-fold
(cross_val_predict), so the fusion model never sees a text score that was fit on
the same row. Val/test text scores come from a model fit on the full train set.

Thresholds are tuned on val for macro-F1; all numbers reported on test.

Output: data/yelp_multimodal_fusion_models/  (text_lr, fusion clf, threshold json)

RUN
    python train_yelp_multimodal_fusion.py
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "yelp_multimodal_features.csv")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "data", "yelp_multimodal_fusion_models")

DROP_FEATS = {"user_reviews_on_this_biz"}   # constant (Yelp = 1 review/user/business)
META_COLS  = {"text", "label", "split", "source"}


def log(m):
    print(m, flush=True)


def best_threshold(y_true, proba):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (proba >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def evaluate(name, y_true, proba, thr, rows):
    pred = (proba >= thr).astype(int)
    rows.append({
        "model": name,
        "thr": round(thr, 2),
        "acc": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "fake_precision": precision_score(y_true, pred, pos_label=1, zero_division=0),
        "fake_recall": recall_score(y_true, pred, pos_label=1, zero_division=0),
    })


def fusion_clf():
    if HAS_XGB:
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                             tree_method="hist", n_jobs=-1, random_state=42)
    from sklearn.ensemble import HistGradientBoostingClassifier
    log("XGBoost not installed — using HistGradientBoosting.")
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=42)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    df["text"] = df["text"].astype(str)
    feat_cols = [c for c in df.columns if c not in META_COLS and c not in DROP_FEATS]
    log(f"rows={len(df):,}  behavioral features={len(feat_cols)}")

    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    y_tr, y_va, y_te = tr.label.values, va.label.values, te.label.values
    log(f"train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    # ----- TEXT modality: P(fake|text), leakage-free on train -------------- #
    log("\nfitting text model (TF-IDF + LR) ...")
    vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True,
                          min_df=2, stop_words="english", dtype=np.float32)
    Xtr_txt = vec.fit_transform(tr.text.values)
    Xva_txt = vec.transform(va.text.values)
    Xte_txt = vec.transform(te.text.values)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # out-of-fold train scores (no row sees a model fit on itself)
        ts_tr = cross_val_predict(lr, Xtr_txt, y_tr, cv=5,
                                  method="predict_proba", n_jobs=-1)[:, 1]
        lr.fit(Xtr_txt, y_tr)
    ts_va = lr.predict_proba(Xva_txt)[:, 1]
    ts_te = lr.predict_proba(Xte_txt)[:, 1]

    # ----- behavioral matrices -------------------------------------------- #
    Btr = tr[feat_cols].values.astype(np.float32)
    Bva = va[feat_cols].values.astype(np.float32)
    Bte = te[feat_cols].values.astype(np.float32)

    rows = []

    # 1) text-only
    thr = best_threshold(y_va, ts_va)
    evaluate("text_only (TF-IDF+LR)", y_te, ts_te, thr, rows)

    # 2) behavioral-only
    log("training behavioral-only ...")
    clf_b = fusion_clf()
    clf_b.fit(Btr, y_tr)
    pb_va = clf_b.predict_proba(Bva)[:, 1]
    pb_te = clf_b.predict_proba(Bte)[:, 1]
    thr = best_threshold(y_va, pb_va)
    evaluate("behavioral_only", y_te, pb_te, thr, rows)

    # 3) FUSION (text score + behavioral)
    log("training FUSION (text + behavioral) ...")
    Ftr = np.hstack([Btr, ts_tr.reshape(-1, 1)])
    Fva = np.hstack([Bva, ts_va.reshape(-1, 1)])
    Fte = np.hstack([Bte, ts_te.reshape(-1, 1)])
    clf_f = fusion_clf()
    clf_f.fit(Ftr, y_tr)
    pf_va = clf_f.predict_proba(Fva)[:, 1]
    pf_te = clf_f.predict_proba(Fte)[:, 1]
    thr_f = best_threshold(y_va, pf_va)
    evaluate("FUSION (multi-modal)", y_te, pf_te, thr_f, rows)

    # ----- report ---------------------------------------------------------- #
    log("\n" + "=" * 78)
    log("YELP MULTI-MODAL FUSION — test-set results (same split as text-only baselines)")
    log("=" * 78)
    log(f"{'model':<26}{'thr':>6}{'acc':>9}{'macroF1':>10}{'fakeP':>9}{'fakeR':>9}")
    log("-" * 78)
    for r in rows:
        log(f"{r['model']:<26}{r['thr']:>6.2f}{r['acc']:>8.1%} {r['macro_f1']:>9.1%} "
            f"{r['fake_precision']:>8.1%} {r['fake_recall']:>8.1%}")
    log("-" * 78)
    lift = rows[2]["macro_f1"] - rows[0]["macro_f1"]
    log(f"multi-modal lift over text-only: {lift:+.1%} macro-F1")

    # feature importances (fusion)
    try:
        names = feat_cols + ["text_score"]
        imp = getattr(clf_f, "feature_importances_", None)
        if imp is not None:
            order = np.argsort(imp)[::-1][:10]
            log("\ntop fusion feature importances:")
            for i in order:
                log(f"  {names[i]:<28}{imp[i]:.4f}")
    except Exception:
        pass

    # ----- save ------------------------------------------------------------ #
    import joblib
    joblib.dump(vec, os.path.join(OUT_DIR, "text_tfidf.joblib"))
    joblib.dump(lr,  os.path.join(OUT_DIR, "text_lr.joblib"))
    joblib.dump(clf_f, os.path.join(OUT_DIR, "fusion_clf.joblib"))
    json.dump({"feat_cols": feat_cols, "threshold": thr_f},
              open(os.path.join(OUT_DIR, "fusion_meta.json"), "w"), indent=2)
    log(f"\nsaved models -> {OUT_DIR}")


if __name__ == "__main__":
    main()
