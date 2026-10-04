"""
merge_all_detectors.py — final 4-detector thesis table. Combines the GPU DL predictions
(dl_predictions_all.csv) with the sklearn members run locally, on the corrected frontend test.

Detectors (each = 8 sklearn + LSTM + DistilBERT, 10-model majority, exactly as served):
  baseline    : baseline sklearn        + base_lstm  + base_db
  focal       : baseline sklearn        + focal_lstm + focal_db
  contrastive : baseline sklearn        + contr_lstm + contr_db
  modelb      : Model B (multimodal) sk + mb_lstm    + mb_db

PREREQ: download dl_predictions_all.csv from compare_dl_all_colab.ipynb into this folder.
RUN: PYTHONIOENCODING=utf-8 python merge_all_detectors.py
"""
import os
os.environ.setdefault("USE_TF", "0")
import json, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score, recall_score, precision_score

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "data")
FRONTEND = os.path.join(DATA, "yelp_frontend_multimodal.csv")
DL = os.path.join(HERE, "dl_predictions_all.csv")
SK = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]
FEAT = ["rating","is_extreme","review_word_len","review_char_len","rating_dev_from_biz",
        "abs_rating_dev_from_biz","user_review_count","user_avg_rating","user_rating_std",
        "user_frac_positive","user_frac_extreme","user_is_singleton","user_max_reviews_per_day",
        "user_reviews_per_day","biz_review_count","biz_avg_rating","biz_rating_std"]
BASE_DIR = os.path.join(DATA, "yelp_fake_sklearn_models")
MB_DIR = os.path.join(DATA, "yelp_multimodal_full_sklearn_models")


def sk_labels(d, X, prefix, thr_file):
    thr = json.load(open(os.path.join(d, thr_file))); lab = {}
    for n in SK:
        p = os.path.join(d, f"{prefix}{n}.joblib")
        if os.path.exists(p):
            lab[n] = (joblib.load(p).predict_proba(X)[:, 1] >= float(thr.get(n, 0.5))).astype(int)
            print(f"    {os.path.basename(d)}/{n} done", flush=True)
    return lab


def main():
    if not os.path.exists(DL):
        raise SystemExit("dl_predictions_all.csv not found — run compare_dl_all_colab.ipynb on GPU and download it here.")
    df = pd.read_csv(FRONTEND); df["review_text"] = df["review_text"].astype(str)
    dl = pd.read_csv(DL)
    assert len(dl) == len(df) and (dl["ground_truth_label"].values == df["ground_truth_label"].values).all(), "align mismatch"
    y = df["ground_truth_label"].values.astype(int)
    texts = df["review_text"].tolist()
    mm = pd.DataFrame({"text": texts})
    for c in FEAT: mm[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")
    print(f"frontend {len(df):,} ({y.mean():.0%} fake)\n", flush=True)

    print("baseline sklearn (text-only) ...", flush=True)
    base_sk = sk_labels(BASE_DIR, texts, "yelp_fake_", "yelp_fake_thresholds.json")
    print("Model B sklearn (multimodal) ...", flush=True)
    mb_sk = sk_labels(MB_DIR, mm, "", "thresholds.json")

    detectors = {
        "baseline":    (base_sk, dl["base_lstm"].values, dl["base_db"].values),
        "focal":       (base_sk, dl["focal_lstm"].values, dl["focal_db"].values),
        "contrastive": (base_sk, dl["contr_lstm"].values, dl["contr_db"].values),
        "modelb":      (mb_sk,   dl["mb_lstm"].values,    dl["mb_db"].values),
    }

    def m(p): return (f1_score(y, p, average="macro", zero_division=0),
                      recall_score(y, p, pos_label=1, zero_division=0),
                      precision_score(y, p, pos_label=1, zero_division=0))
    print("\n" + "=" * 78)
    print(f"CLEAN current-split 4-detector comparison — frontend test (n={len(y):,})")
    print("=" * 78)
    print(f"{'detector':<13}{'ENSEMBLE(10)':>16}{'LSTM':>9}{'DistilBERT':>12}{'best sklearn':>14}")
    print("-" * 78)
    for name, (sk, ls, db) in detectors.items():
        members = {**sk, "lstm": ls, "distilbert": db}
        M = np.vstack([np.asarray(members[k], int) for k in members])
        ens = (M.sum(0) > M.shape[0] / 2).astype(int)
        best_sk = max(((n, m(sk[n])[0]) for n in sk), key=lambda x: x[1])
        e = m(ens); l = m(ls); d_ = m(db)
        print(f"{name:<13}{f'{e[0]:.1%}/{e[1]:.0%}/{e[2]:.0%}':>16}{l[0]:>8.1%}{d_[0]:>12.1%}"
              f"{f'{best_sk[0]} {best_sk[1]:.1%}':>14}")
    print("=" * 78)
    print("(ENSEMBLE = macroF1/fakeRecall/fakePrecision, 10-model majority as served)")


if __name__ == "__main__":
    main()
