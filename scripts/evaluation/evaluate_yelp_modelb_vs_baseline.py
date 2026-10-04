# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
evaluate_yelp_modelb_vs_baseline.py — the honest, apples-to-apples comparison.

Runs BOTH the served multi-modal **Model B** ensemble AND the text-only **baseline**
ensemble on the *identical* clean Yelp test reviews, and prints per-model + majority-
ensemble macro-F1 side by side.

Why this is the fair test (and why the existing numbers weren't directly comparable):
  * Test set = `yelp_multimodal_features.csv` split=="test" — the SAME held-out Yelp
    reviews for both models (genuine + HUMAN fakes only; NO injected AI fakes), so
    "Fake recall" here is true human-fake recall. Leakage-free: it's the shared
    yelp_split test split, which neither model trained on.
  * Model B's previously-reported 74-77% came from its training script on the
    AI-AUGMENTED test set (AI fakes inflate it) — not comparable to the baseline's 67%.
    This script removes that confound by scoring the served Model B on the clean test.

Both families are scored through their canonical code paths (val-tuned thresholds):
  baseline → score_reviews._run_yelp_* ; Model B → model_b._score_*.

Usage:
    python evaluate_yelp_modelb_vs_baseline.py            # full test set (~18k, slow: ~15-30 min CPU)
    python evaluate_yelp_modelb_vs_baseline.py 3000       # quick representative sample
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

import model_b
from score_reviews import _run_yelp_sklearn_models, _run_yelp_lstm, _run_yelp_distilbert

FEATURES = os.path.join(os.path.dirname(__file__), "data", "yelp_multimodal_features.csv")
SK_NAMES = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]
ALL      = SK_NAMES + ["lstm", "distilbert"]
CHUNK    = 1000   # process in chunks (bounds memory; KNN OOMs on the full set at once)

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else None


def _labels(per_text_dicts, name=None):
    """Pull the label array out of a list of per-text result dicts."""
    if name is None:   # the dict IS the model result (lstm/distilbert)
        return [(d or {}).get("label") for d in per_text_dicts]
    return [((d or {}).get(name, {}) or {}).get("label") for d in per_text_dicts]


def main():
    print("Loading clean Yelp test split (same reviews for both models)...")
    df = pd.read_csv(FEATURES)
    te = df[df.split == "test"].reset_index(drop=True)
    if SAMPLE:
        te = te.sample(min(SAMPLE, len(te)), random_state=42).reset_index(drop=True)
    texts = te["text"].astype(str).tolist()
    y = te["label"].values.astype(int)
    feat_records = te[model_b.FEAT_COLS].to_dict("records")
    reviews = [{"review_text": texts[i], **feat_records[i]} for i in range(len(te))]
    print(f"  test rows: {len(texts):,}  (genuine={int((y==0).sum()):,}  human-fake={int((y==1).sum()):,})")
    print(f"  scoring 10 baseline + 10 Model-B models in chunks of {CHUNK} ... (DistilBERT is the slow part)\n")

    base = {n: [] for n in ALL}     # name -> list of labels
    mb   = {n: [] for n in ALL}

    for s in range(0, len(texts), CHUNK):
        ct, cr = texts[s:s + CHUNK], reviews[s:s + CHUNK]
        # baseline (text-only)
        b_sk = _run_yelp_sklearn_models(ct)
        b_ls = _run_yelp_lstm(ct, "baseline")
        b_bt = _run_yelp_distilbert(ct, "baseline")
        for n in SK_NAMES: base[n] += _labels(b_sk, n)
        base["lstm"] += _labels(b_ls); base["distilbert"] += _labels(b_bt)
        # Model B (multi-modal)
        m_sk = model_b._score_sklearn(cr)
        m_ls = model_b._score_lstm(cr)
        m_db = model_b._score_db(cr)
        for n in SK_NAMES: mb[n] += _labels(m_sk, n)
        mb["lstm"] += _labels(m_ls); mb["distilbert"] += _labels(m_db)
        print(f"  scored {min(s+CHUNK, len(texts)):,}/{len(texts):,}", flush=True)

    def m(pred):
        p = np.array([(-1 if v is None else v) for v in pred])
        mask = p >= 0
        if mask.sum() == 0:
            return None
        yt, yp = y[mask], p[mask]
        return (f1_score(yt, yp, average="macro", zero_division=0),
                recall_score(yt, yp, pos_label=1, zero_division=0),
                precision_score(yt, yp, pos_label=1, zero_division=0))

    def majority(fam):
        out = []
        for i in range(len(y)):
            votes = [fam[n][i] for n in ALL if fam[n][i] is not None]
            out.append(1 if votes and sum(votes) > len(votes) / 2 else 0)
        return out

    # ── table ──
    print("\n" + "=" * 78)
    print(f"YELP - MODEL B (multi-modal) vs TEXT-ONLY BASELINE  |  clean test, n={len(y):,}")
    print("  (Fake recall = HUMAN-fake recall - no AI fakes in this set)")
    print("=" * 78)
    print(f"{'model':<12}{'BASELINE macroF1 / fakeR / fakeP':>33}   {'MODEL B macroF1 / fakeR / fakeP':>33}")
    print("-" * 78)

    def fmt(t):
        return "  unavailable" if t is None else f"{t[0]:6.1%} {t[1]:6.1%} {t[2]:6.1%}"

    for n in ALL:
        print(f"{n:<12}{fmt(m(base[n])):>33}   {fmt(m(mb[n])):>33}")

    be, me = m(majority(base)), m(majority(mb))
    print("-" * 78)
    print(f"{'ENSEMBLE':<12}{fmt(be):>33}   {fmt(me):>33}   (majority vote of the 10)")
    print("=" * 78)

    if be and me:
        d = (me[0] - be[0]) * 100
        verdict = ("Model B WINS" if d > 0.5 else "BASELINE wins" if d < -0.5 else "~ TIE")
        print(f"\nEnsemble macro-F1: baseline {be[0]:.1%}  vs  Model B {me[0]:.1%}   "
              f"(delta {d:+.1f} pp -> {verdict})")
        print(f"Human-fake recall: baseline {be[1]:.1%}  vs  Model B {me[1]:.1%}   "
              f"(delta {(me[1]-be[1])*100:+.1f} pp)")
        print("\nThis is the apples-to-apples number for the thesis: does multi-modal fusion")
        print("beat text-only on the SAME clean Yelp reviews, using the DEPLOYED models?")


if __name__ == "__main__":
    main()
