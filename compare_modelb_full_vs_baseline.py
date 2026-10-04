"""
compare_modelb_full_vs_baseline.py — re-run the frontend-business comparison with the NEW
pure-full Model B.

Earlier (`evaluate_yelp_modelb_vs_baseline.py`) we found the OLD (AI-augmented) Model B scored
BELOW the text-only baseline on the clean Yelp reviews served on the frontend — which is why we
considered replacing it with text-only. Now we re-run the SAME apples-to-apples test with THREE
arms on the exact reviews the frontend serves (`data/yelp_frontend_multimodal.csv`, 17,891 rows
with `ground_truth_label` + the 17 behavioral features):

  * BASELINE   — text-only ensemble (score_reviews._run_yelp_*)               [the replacement candidate]
  * OLD MODEL B — AI-augmented multimodal ensemble (data/yelp_multimodal_*_onnx, *_sklearn_models)
  * NEW MODEL B — pure-full multimodal ensemble (data/yelp_multimodal_full_sklearn_models +
                  yelp_multimodal_lstm_full_onnx + yelp_multimodal_distilbert_full_onnx)

All scored through their canonical code paths with val-tuned thresholds. Reports per-model and
majority-ensemble macro-F1 / fake recall / fake precision, and the verdict: does the NEW Model B
finally beat text-only on the frontend businesses?

Usage:
    python compare_modelb_full_vs_baseline.py 4000     # representative sample (default)
    python compare_modelb_full_vs_baseline.py 0        # full 17,891 (slow: DistilBERT x3 arms)
"""

import os
os.environ.setdefault("USE_TF", "0")          # stop transformers/optimum importing TensorFlow
os.environ.setdefault("USE_TORCH", "1")        # (protobuf gencode clash) -> baseline DistilBERT loads
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

import model_b
from score_reviews import _run_yelp_sklearn_models, _run_yelp_lstm, _run_yelp_distilbert

HERE = os.path.dirname(__file__)
FRONTEND = os.path.join(HERE, "data", "yelp_frontend_multimodal.csv")
SK_NAMES = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]
ALL = SK_NAMES + ["lstm", "distilbert"]
CHUNK = 1000

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 4000


def _lab(dicts, name=None):
    if name is None:
        return [(d or {}).get("label") for d in dicts]
    return [((d or {}).get(name, {}) or {}).get("label") for d in dicts]


def score_modelb(reviews):
    """Score the currently-configured Model B (dirs set on model_b) over all reviews, chunked."""
    fam = {n: [] for n in ALL}
    for s in range(0, len(reviews), CHUNK):
        cr = reviews[s:s + CHUNK]
        sk = model_b._score_sklearn(cr); ls = model_b._score_lstm(cr); db = model_b._score_db(cr)
        for n in SK_NAMES:
            fam[n] += _lab(sk, n)
        fam["lstm"] += _lab(ls); fam["distilbert"] += _lab(db)
        print(f"    scored {min(s+CHUNK, len(reviews)):,}/{len(reviews):,}", flush=True)
    return fam


def repoint_modelb_to_full():
    """Repoint model_b at the pure-full dirs and clear its caches so it reloads them."""
    model_b.SK_DIR   = os.path.join(HERE, "data", "yelp_multimodal_full_sklearn_models")
    model_b.LSTM_DIR = os.path.join(HERE, "data", "yelp_multimodal_lstm_full_onnx")
    model_b.DB_DIR   = os.path.join(HERE, "data", "yelp_multimodal_distilbert_full_onnx")
    model_b._sk = None; model_b._sk_thr = None; model_b._lstm = None; model_b._db = None


def main():
    df = pd.read_csv(FRONTEND)
    df["review_text"] = df["review_text"].astype(str)
    if SAMPLE and SAMPLE > 0:
        df = df.sample(min(SAMPLE, len(df)), random_state=42).reset_index(drop=True)
    y = df["ground_truth_label"].values.astype(int)
    texts = df["review_text"].tolist()
    feat_records = df[model_b.FEAT_COLS].to_dict("records")
    reviews = [{"review_text": texts[i], **feat_records[i]} for i in range(len(df))]
    print(f"Frontend businesses: {len(df):,} reviews  "
          f"(genuine={int((y==0).sum()):,}  fake={int((y==1).sum()):,}, {y.mean():.1%} fake)\n")

    # 1) text-only baseline
    print("[1/3] text-only BASELINE ...", flush=True)
    base = {n: [] for n in ALL}
    for s in range(0, len(texts), CHUNK):
        ct = texts[s:s + CHUNK]
        b_sk = _run_yelp_sklearn_models(ct)
        for n in SK_NAMES:
            base[n] += _lab(b_sk, n)
        base["lstm"] += _lab(_run_yelp_lstm(ct, "baseline"))
        base["distilbert"] += _lab(_run_yelp_distilbert(ct, "baseline"))
        print(f"    scored {min(s+CHUNK, len(texts)):,}/{len(texts):,}", flush=True)

    # 2) OLD Model B (default AI-augmented dirs)
    print("[2/3] OLD Model B (AI-augmented) ...", flush=True)
    old = score_modelb(reviews)

    # 3) NEW pure-full Model B
    print("[3/3] NEW Model B (pure-full) ...", flush=True)
    repoint_modelb_to_full()
    new = score_modelb(reviews)

    def trip(pred):
        p = np.array([(-1 if v is None else v) for v in pred]); mask = p >= 0
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

    def cell(t):
        return "   n/a" if t is None else f"{t[0]:5.1%}/{t[1]:5.1%}/{t[2]:5.1%}"

    print("\n" + "=" * 86)
    print(f"FRONTEND BUSINESSES — macroF1 / fakeRecall / fakePrecision   (n={len(y):,})")
    print("=" * 86)
    print(f"{'model':<12}{'TEXT-ONLY':>20}{'OLD Model B':>20}{'NEW Model B (full)':>22}")
    print("-" * 86)
    for n in ALL:
        print(f"{n:<12}{cell(trip(base[n])):>20}{cell(trip(old[n])):>20}{cell(trip(new[n])):>22}")
    be, oe, ne = trip(majority(base)), trip(majority(old)), trip(majority(new))
    print("-" * 86)
    print(f"{'ENSEMBLE':<12}{cell(be):>20}{cell(oe):>20}{cell(ne):>22}   (majority of 10)")
    print("=" * 86)

    print("\nEnsemble macro-F1:")
    print(f"  text-only baseline : {be[0]:.1%}")
    print(f"  OLD Model B        : {oe[0]:.1%}   ({(oe[0]-be[0])*100:+.1f} pp vs baseline)")
    print(f"  NEW Model B (full) : {ne[0]:.1%}   ({(ne[0]-be[0])*100:+.1f} pp vs baseline)")
    d = (ne[0] - be[0]) * 100
    verdict = ("NEW Model B WINS" if d > 0.5 else "BASELINE wins" if d < -0.5 else "~ TIE")
    print(f"\nVERDICT on the frontend businesses: {verdict} (NEW Model B {d:+.1f} pp vs text-only)")


if __name__ == "__main__":
    main()
