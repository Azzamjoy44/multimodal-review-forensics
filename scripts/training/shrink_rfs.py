# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
shrink_rfs.py — retrain the two remaining bloated Random Forests with bounded trees,
the same fix already applied to the sentiment RF (shrink_sentiment_rf.py):

    * Review/Non-review detector RF   data/review_detector_sklearn_models/rf.joblib  (~797 MB)
    * Multi-modal Yelp (Model B) RF    data/yelp_multimodal_sklearn_models/rf.joblib   (~531 MB)

Both were trained with unbounded depth -> trees grow until pure -> huge models. Capping
max_depth + min_samples_leaf shrinks them with negligible accuracy change.

This retrains ONLY the RF for each (not the other 7 models) inside the EXACT pipeline +
threshold logic from the original training scripts (imported, not re-implemented), so
the result is consistent: it re-tunes the RF's decision threshold on the val split
(macro-F1) and updates only the "rf" entry in each thresholds.json. The other models'
thresholds are preserved. Old rf.joblib files are backed up to .bak.

Run:  python shrink_rfs.py
"""

import os
import json
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

import train_review_detector_sklearn as rd      # has __main__ guard — safe to import
import train_yelp_multimodal_sklearn as mm

# ── tune these to trade size vs accuracy (same as the sentiment fix) ──
MAX_DEPTH        = 40
MIN_SAMPLES_LEAF = 5


def _fmt(b):
    return f"{b/1e9:.2f} GB" if b >= 1e9 else f"{b/1e6:.0f} MB"


def _save_rf(pipe, model_dir, thr):
    """Back up the old rf.joblib, save the new one, update only thresholds.json['rf']."""
    path = os.path.join(model_dir, "rf.joblib")
    old = os.path.getsize(path) if os.path.exists(path) else 0
    if old:
        bak = path + ".bak"
        if os.path.exists(bak):
            os.remove(path)
        else:
            os.replace(path, bak)
            print(f"  backed up old model -> {os.path.basename(bak)}")
    joblib.dump(pipe, path)
    new = os.path.getsize(path)
    tp = os.path.join(model_dir, "thresholds.json")
    thresholds = json.load(open(tp)) if os.path.exists(tp) else {}
    thresholds["rf"] = thr
    json.dump(thresholds, open(tp, "w"), indent=2)
    ratio = f"  ({old/new:.0f}x smaller)" if old and new else ""
    print(f"  rf.joblib: {_fmt(old) if old else '?'} -> {_fmt(new)}{ratio}   thr={thr:.2f} (updated in thresholds.json)")


def _rf():
    return RandomForestClassifier(n_estimators=100, max_depth=MAX_DEPTH,
                                  min_samples_leaf=MIN_SAMPLES_LEAF, class_weight="balanced",
                                  random_state=42, n_jobs=-1)


def shrink_review_detector():
    print("\n=== Review/Non-review detector RF ===")
    df = pd.read_csv(rd.DATA_FILE); df["text"] = df["text"].astype(str)
    tr = df[df.split == "train"]; va = df[df.split == "val"]; te = df[df.split == "test"]
    y_tr, y_va, y_te = tr["label"].values, va["label"].values, te["label"].values

    vec = rd.make_vectorizer(5_000)              # exact same vectorizer config as training
    Xtr = vec.fit_transform(tr["text"].values)
    Xva = vec.transform(va["text"].values)
    Xte = vec.transform(te["text"].values)

    clf = _rf()
    t = time.time(); clf.fit(Xtr, y_tr); print(f"  fit in {time.time()-t:.0f}s")
    thr = rd.best_threshold(y_va, clf.predict_proba(Xva)[:, 1])   # re-tune threshold on val
    pred = (clf.predict_proba(Xte)[:, 1] >= thr).astype(int)
    print(f"  TEST  acc={accuracy_score(y_te, pred):.3%}  "
          f"macroF1={f1_score(y_te, pred, average='macro', zero_division=0):.3%}")
    _save_rf(Pipeline([("tfidf", vec), ("clf", clf)]), rd.MODEL_DIR, thr)


def shrink_model_b():
    print("\n=== Multi-modal Yelp (Model B) RF ===")
    df = pd.read_csv(mm.DATA_FILE); df["text"] = df["text"].astype(str)
    feat_cols = [c for c in df.columns if c not in mm.META and c not in mm.DROP]
    tr = df[df.split == "train"]; va = df[df.split == "val"]; te = df[df.split == "test"]
    y_tr, y_va, y_te = tr.label.values, va.label.values, te.label.values
    cols = ["text"] + feat_cols
    Xtr, Xva, Xte = tr[cols], va[cols], te[cols]
    sw = np.where(tr["fake_type"].values == "ai_fake", mm.AI_WEIGHT, 1.0)   # AI up-weighting

    pipe = mm.make_pipeline(_rf(), 5_000, feat_cols)   # exact ColumnTransformer + clf
    t = time.time(); pipe.fit(Xtr, y_tr, clf__sample_weight=sw); print(f"  fit in {time.time()-t:.0f}s")
    thr = mm.best_threshold(y_va, pipe.predict_proba(Xva)[:, 1])
    pred = (pipe.predict_proba(Xte)[:, 1] >= thr).astype(int)
    ft = te["fake_type"].values
    def rec(m): return float(pred[m].mean()) if m.sum() else float("nan")
    print(f"  TEST  macroF1={f1_score(y_te, pred, average='macro', zero_division=0):.3%}  "
          f"humanFakeR={rec(ft=='human_fake'):.1%}  aiFakeR={rec(ft=='ai_fake'):.1%}  "
          f"genuineSpec={1.0-rec(ft=='genuine'):.1%}")
    _save_rf(pipe, mm.OUT_DIR, thr)


def main():
    shrink_review_detector()
    shrink_model_b()
    print("\nDone. Restart the server to load the smaller models. After confirming the")
    print("metrics look fine, delete the two rf.joblib.bak files to reclaim the disk.")


if __name__ == "__main__":
    main()
