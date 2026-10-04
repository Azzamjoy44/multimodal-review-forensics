"""
per_business_winrate.py — per-business accuracy of Model B vs the text-only baseline, so we can
pick businesses to eyeball on the frontend. Uses the served ensemble members (7 sklearn ex-knn
for speed + LSTM + DistilBERT from dl_predictions_all.csv). Reports Model B's win-rate across
businesses and lists the biggest wins / losses (filtered to businesses with enough reviews).
RUN: PYTHONIOENCODING=utf-8 python per_business_winrate.py
"""
import os
os.environ.setdefault("USE_TF", "0")
import json, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "data")
SK = ["rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]          # knn skipped (slow + weak)
FEAT = ["rating","is_extreme","review_word_len","review_char_len","rating_dev_from_biz",
        "abs_rating_dev_from_biz","user_review_count","user_avg_rating","user_rating_std",
        "user_frac_positive","user_frac_extreme","user_is_singleton","user_max_reviews_per_day",
        "user_reviews_per_day","biz_review_count","biz_avg_rating","biz_rating_std"]
MIN_REVIEWS = 40


def sk_votes(d, X, prefix, thr_file):
    thr = json.load(open(os.path.join(d, thr_file))); V = []
    for n in SK:
        p = os.path.join(d, f"{prefix}{n}.joblib")
        if os.path.exists(p):
            V.append((joblib.load(p).predict_proba(X)[:, 1] >= float(thr.get(n, 0.5))).astype(int))
            print(f"    {os.path.basename(d)}/{n}", flush=True)
    return np.vstack(V)


def main():
    df = pd.read_csv(os.path.join(DATA, "yelp_frontend_multimodal.csv")); df["review_text"] = df["review_text"].astype(str)
    dl = pd.read_csv(os.path.join(HERE, "dl_predictions_all.csv"))
    assert len(dl) == len(df) and (dl["ground_truth_label"].values == df["ground_truth_label"].values).all()
    y = df["ground_truth_label"].values.astype(int)
    texts = df["review_text"].tolist()
    mm = pd.DataFrame({"text": texts})
    for c in FEAT: mm[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")

    print("baseline sklearn ...", flush=True)
    base_sk = sk_votes(os.path.join(DATA, "yelp_fake_sklearn_models"), texts, "yelp_fake_", "yelp_fake_thresholds.json")
    print("Model B sklearn ...", flush=True)
    mb_sk = sk_votes(os.path.join(DATA, "yelp_multimodal_full_sklearn_models"), mm, "", "thresholds.json")

    # 9-model ensembles (7 sklearn + 2 DL)
    base_M = np.vstack([base_sk, dl["base_lstm"].values, dl["base_db"].values])
    mb_M   = np.vstack([mb_sk,   dl["mb_lstm"].values,   dl["mb_db"].values])
    base_pred = (base_M.sum(0) > base_M.shape[0] / 2).astype(int)
    mb_pred   = (mb_M.sum(0)   > mb_M.shape[0]   / 2).astype(int)

    g = pd.DataFrame({"biz": df["business_id"].values, "y": y,
                      "base_ok": (base_pred == y).astype(int), "mb_ok": (mb_pred == y).astype(int)})
    agg = g.groupby("biz").agg(n=("y", "size"), base=("base_ok", "mean"), mb=("mb_ok", "mean")).reset_index()
    agg = agg[agg.n >= MIN_REVIEWS].copy()
    agg["margin"] = agg["mb"] - agg["base"]
    agg = agg.sort_values("margin", ascending=False)

    win = (agg.margin > 0).mean(); tie = (agg.margin == 0).mean()
    print("\n" + "=" * 64)
    print(f"Per-business (>= {MIN_REVIEWS} reviews): {len(agg)} businesses")
    print(f"  Model B beats baseline: {win:.0%}  | ties: {tie:.0%}  | loses: {1-win-tie:.0%}")
    print("=" * 64)
    def show(rows, title):
        print(f"\n{title}")
        print(f"  {'business':<14}{'n':>5}{'baseline':>10}{'ModelB':>9}{'margin':>9}")
        for _, r in rows.iterrows():
            print(f"  {r.biz:<14}{int(r.n):>5}{r.base:>9.0%}{r.mb:>9.0%}{r.margin*100:>+8.0f}")
    show(agg.head(12), "BIGGEST MODEL B WINS (test these):")
    show(agg[abs(agg.margin) < 0.02].head(4), "~TIES:")
    show(agg.tail(5), "BASELINE WINS (Model B loses — for honesty, e.g. zip_3237):")


if __name__ == "__main__":
    main()
