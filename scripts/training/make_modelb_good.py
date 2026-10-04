# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
make_modelb_good.py — does a PROPERLY-built fusion (strong text + behavioral) beat
text-only on Yelp? The clean, leakage-free stacking test.

Every prior fusion was handicapped: the served Model B was trained on the AI-augmented
objective (x10 AI weight), and the "feature-level fusion" used a single weak LR text
branch (~67%) and only ever compared to that same weak LR. Nobody fused behavioral onto
the STRONG text signal (the deployed 8 sklearn + LSTM + DistilBERT ensemble, ~75-82%).

This does exactly that. We stack the full text-only ensemble's per-model probabilities
and learn a meta-classifier in two forms that isolate the behavioral contribution:
    text-stack : meta-clf on the 10 text probs only          (best text-only can do)
    full-stack : meta-clf on the 10 text probs + 17 behavioral (adds behavioral)
full-stack has strictly more information, so it should be >= text-stack — the whole point.

Leakage-free: text models were trained on TRAIN; we take their clean probs on VAL+TEST,
train the meta-clf on VAL (threshold via OOF cross_val_predict), evaluate on TEST.

RUN: python make_modelb_good.py [val_n] [test_n]     (default 5000 5000)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")

import model_b   # FEAT_COLS
from score_reviews import _run_yelp_sklearn_models, _run_yelp_lstm, _run_yelp_distilbert

FEATURES = os.path.join(os.path.dirname(__file__), "data", "yelp_multimodal_features.csv")
SK_NAMES = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]
TEXT_MODELS = SK_NAMES + ["lstm", "distilbert"]
CHUNK = 1000

VAL_N  = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
TEST_N = int(sys.argv[2]) if len(sys.argv) > 2 else 5000


def _prob(label, conf):
    """Recover P(fake) from a text model's (label, confidence%) verdict."""
    if label is None or conf is None:
        return 0.5
    return conf / 100.0 if label == 1 else 1.0 - conf / 100.0


def text_probs(texts):
    """[N, 10] P(fake) from the deployed text-only models (8 sklearn + lstm + distilbert)."""
    P = np.full((len(texts), len(TEXT_MODELS)), 0.5, dtype=np.float32)
    for s in range(0, len(texts), CHUNK):
        ct = texts[s:s + CHUNK]
        sk = _run_yelp_sklearn_models(ct)
        ls = _run_yelp_lstm(ct, "baseline")
        bt = _run_yelp_distilbert(ct, "baseline")
        for i in range(len(ct)):
            for j, n in enumerate(SK_NAMES):
                d = (sk[i].get(n) or {})
                P[s + i, j] = _prob(d.get("label"), d.get("confidence"))
            P[s + i, 8] = _prob(ls[i].get("label"), ls[i].get("confidence"))
            P[s + i, 9] = _prob(bt[i].get("label"), bt[i].get("confidence"))
        print(f"  text-scored {min(s + CHUNK, len(texts)):,}/{len(texts):,}", flush=True)
    return P


def best_threshold(y, proba):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y, (proba >= t).astype(int), average="macro", zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


def metrics(y, proba, thr):
    pred = (proba >= thr).astype(int)
    return (f1_score(y, pred, average="macro", zero_division=0),
            recall_score(y, pred, pos_label=1, zero_division=0),
            precision_score(y, pred, pos_label=1, zero_division=0))


def make_clf():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                             tree_method="hist", n_jobs=-1, random_state=42)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)


def stack(name, Xva, yva, Xte, yte, rows):
    """Train meta-clf on VAL (OOF threshold), evaluate on TEST."""
    clf = make_clf()
    oof = cross_val_predict(clf, Xva, yva, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
    thr = best_threshold(yva, oof)
    clf.fit(Xva, yva)
    pte = clf.predict_proba(Xte)[:, 1]
    mf1, fr, fp = metrics(yte, pte, thr)
    rows.append((name, thr, mf1, fr, fp))
    return clf


def main():
    df = pd.read_csv(FEATURES)
    df["text"] = df["text"].astype(str)
    va = df[df.split == "val"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    if VAL_N:  va = va.sample(min(VAL_N, len(va)), random_state=42).reset_index(drop=True)
    if TEST_N: te = te.sample(min(TEST_N, len(te)), random_state=42).reset_index(drop=True)
    yva, yte = va.label.values.astype(int), te.label.values.astype(int)
    Bva = va[model_b.FEAT_COLS].values.astype(np.float32)
    Bte = te[model_b.FEAT_COLS].values.astype(np.float32)
    print(f"val={len(va):,} (fake {int(yva.sum()):,}) | test={len(te):,} (fake {int(yte.sum()):,})")

    print("\nscoring text-only models on VAL ...");  Tva = text_probs(va.text.tolist())
    print("scoring text-only models on TEST ...");   Tte = text_probs(te.text.tolist())

    rows = []
    # text-only references on TEST (threshold tuned on val) ----------------------
    di = TEXT_MODELS.index("distilbert")
    thr = best_threshold(yva, Tva[:, di])
    rows.append(("text: DistilBERT only", thr, *metrics(yte, Tte[:, di], thr)))
    ens_va, ens_te = Tva.mean(1), Tte.mean(1)              # mean-prob text ensemble
    thr = best_threshold(yva, ens_va)
    rows.append(("text: ensemble (mean)", thr, *metrics(yte, ens_te, thr)))

    # behavioral-only -----------------------------------------------------------
    stack("behavioral-only (17 feat)", Bva, yva, Bte, yte, rows)
    # the two key stacks --------------------------------------------------------
    stack("text-stack (10 probs)", Tva, yva, Tte, yte, rows)
    fic = stack("FULL-stack (10 probs + 17 beh)",
                np.hstack([Tva, Bva]), yva, np.hstack([Tte, Bte]), yte, rows)

    print("\n" + "=" * 76)
    print(f"MAKE MODEL B GOOD — leakage-free stacking on clean Yelp test (n={len(yte):,})")
    print("=" * 76)
    print(f"{'model':<32}{'thr':>6}{'macroF1':>10}{'fakeR':>8}{'fakeP':>8}")
    print("-" * 76)
    for name, t, mf1, fr, fp in rows:
        print(f"{name:<32}{t:>6.2f}{mf1:>9.1%}{fr:>8.1%}{fp:>8.1%}")
    print("-" * 76)
    ts = next(r for r in rows if r[0].startswith("text-stack"))
    fs = next(r for r in rows if r[0].startswith("FULL"))
    d = (fs[2] - ts[2]) * 100
    print(f"behavioral lift on top of the strong text signal: {d:+.1f} pp macro-F1  "
          f"({'REAL — behavioral helps' if d > 0.5 else 'none — text ceiling' if d < 0.5 else 'marginal'})")

    try:
        imp = getattr(fic, "feature_importances_", None)
        if imp is not None:
            names = [f"txt:{n}" for n in TEXT_MODELS] + model_b.FEAT_COLS
            order = np.argsort(imp)[::-1][:12]
            print("\ntop FULL-stack feature importances:")
            for i in order:
                print(f"  {names[i]:<28}{imp[i]:.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
