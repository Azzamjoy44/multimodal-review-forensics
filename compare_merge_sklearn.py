"""
compare_merge_sklearn.py — final step of the split (Colab DL + local sklearn) frontend
comparison. Scores the sklearn members of each arm LOCALLY (DIRECT predict_proba — no KNN,
no reason extraction, so it's fast), merges the DL labels from Colab (`dl_predictions.csv`),
and prints the per-model + majority-ensemble macro-F1 table for: text-only baseline / OLD
Model B / NEW pure-full Model B.

KNN is intentionally skipped: it's the slowest member (brute cosine over the train set) and
weakest (~66%); excluding 1 of 10 weak votes is negligible for the ensemble, and its
standalone number is already on record from the model evals.

PREREQ: download dl_predictions.csv from compare_dl_colab.ipynb into this folder.
RUN: python compare_merge_sklearn.py
"""
import os
os.environ.setdefault("USE_TF", "0")
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, recall_score, precision_score
warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
FRONTEND = os.path.join(DATA, "yelp_frontend_multimodal.csv")
DL = os.path.join(HERE, "dl_predictions.csv")
SK = ["rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]          # knn skipped (slow+weak)
ALL = SK + ["lstm", "distilbert"]
FEAT = ["rating", "is_extreme", "review_word_len", "review_char_len", "rating_dev_from_biz",
        "abs_rating_dev_from_biz", "user_review_count", "user_avg_rating", "user_rating_std",
        "user_frac_positive", "user_frac_extreme", "user_is_singleton", "user_max_reviews_per_day",
        "user_reviews_per_day", "biz_review_count", "biz_avg_rating", "biz_rating_std"]

BASE_DIR = os.path.join(DATA, "yelp_fake_sklearn_models")            # text-only baseline
OLD_DIR  = os.path.join(DATA, "yelp_multimodal_sklearn_models")      # OLD Model B (AI-aug)
NEW_DIR  = os.path.join(DATA, "yelp_multimodal_full_sklearn_models") # NEW pure-full Model B


def score_dir(d, X, prefix="", thr_file="thresholds.json"):
    """X = list of texts (text-only dir) OR DataFrame text+FEAT (multimodal dir)."""
    thr = json.load(open(os.path.join(d, thr_file)))
    fam = {}
    for n in SK:
        p = os.path.join(d, f"{prefix}{n}.joblib")
        if not os.path.exists(p):
            continue
        proba = joblib.load(p).predict_proba(X)[:, 1]
        fam[n] = (proba >= float(thr.get(n, 0.5))).astype(int)
        print(f"    {os.path.basename(d)}/{n} done", flush=True)
    return fam


def main():
    if not os.path.exists(DL):
        raise SystemExit("dl_predictions.csv not found — run compare_dl_colab.ipynb and download it here first.")
    df = pd.read_csv(FRONTEND); df["review_text"] = df["review_text"].astype(str)
    dl = pd.read_csv(DL)
    assert len(dl) == len(df), f"row mismatch: dl {len(dl)} vs frontend {len(df)}"
    y = df["ground_truth_label"].values.astype(int)
    assert (dl["ground_truth_label"].values == y).all(), "label/order mismatch between dl and frontend"

    texts = df["review_text"].tolist()
    mm = pd.DataFrame({"text": texts})
    for c in FEAT:
        mm[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float32")
    print(f"frontend: {len(df):,} reviews ({y.mean():.1%} fake)\n", flush=True)

    print("[1/3] baseline sklearn (text-only) ...", flush=True)
    base = score_dir(BASE_DIR, texts, prefix="yelp_fake_", thr_file="yelp_fake_thresholds.json")
    print("[2/3] OLD Model B sklearn ...", flush=True);            old = score_dir(OLD_DIR, mm)
    print("[3/3] NEW Model B sklearn ...", flush=True);            new = score_dir(NEW_DIR, mm)

    base["lstm"], base["distilbert"] = dl["base_lstm"].values, dl["base_db"].values
    old["lstm"], old["distilbert"]   = dl["old_lstm"].values, dl["old_db"].values
    new["lstm"], new["distilbert"]   = dl["new_lstm"].values, dl["new_db"].values

    def trip(pred):
        p = np.asarray(pred, dtype=int)
        return (f1_score(y, p, average="macro", zero_division=0),
                recall_score(y, p, pos_label=1, zero_division=0),
                precision_score(y, p, pos_label=1, zero_division=0))

    def majority(fam):
        M = np.vstack([np.asarray(fam[n], dtype=int) for n in ALL if n in fam])
        return (M.sum(0) > M.shape[0] / 2).astype(int)

    def cell(t): return f"{t[0]:5.1%}/{t[1]:5.1%}/{t[2]:5.1%}"
    print("\n" + "=" * 86)
    print(f"FRONTEND BUSINESSES (corrected features) — macroF1 / fakeR / fakeP   (n={len(y):,})")
    print("=" * 86)
    print(f"{'model':<12}{'TEXT-ONLY':>20}{'OLD Model B':>20}{'NEW Model B (full)':>22}")
    print("-" * 86)
    for n in ALL:
        print(f"{n:<12}{cell(trip(base[n])):>20}{cell(trip(old[n])):>20}{cell(trip(new[n])):>22}")
    be, oe, ne = trip(majority(base)), trip(majority(old)), trip(majority(new))
    print("-" * 86)
    print(f"{'ENSEMBLE':<12}{cell(be):>20}{cell(oe):>20}{cell(ne):>22}   (majority of 9)")
    print("=" * 86)
    d = (ne[0] - be[0]) * 100
    print(f"\nEnsemble macro-F1:  text-only {be[0]:.1%}  |  OLD MB {oe[0]:.1%} ({(oe[0]-be[0])*100:+.1f})  |  "
          f"NEW MB {ne[0]:.1%} ({d:+.1f} vs baseline)")
    print("VERDICT:", "NEW Model B WINS" if d > 0.5 else "BASELINE wins" if d < -0.5 else "~TIE",
          f"(NEW MB {d:+.1f} pp vs text-only)")


if __name__ == "__main__":
    main()
