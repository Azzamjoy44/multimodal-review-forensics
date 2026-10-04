# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
per_business_full.py — the COMPLETE per-business win/tie/lose list of Model B vs the text-only
baseline (9-model ensembles: 7 sklearn ex-knn + LSTM + DistilBERT from dl_predictions_all.csv),
for every business with >= MIN_REVIEWS served test reviews. Prints all three groups in full and
writes per_business_full.csv. RUN: PYTHONIOENCODING=utf-8 python -u per_business_full.py
"""
import os
os.environ.setdefault("USE_TF", "0")
import json, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
import model_b  # reuse FEAT_COLS only

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
SK = ["rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]      # knn skipped (slow + weak)
FEAT = model_b.FEAT_COLS
MIN_REVIEWS = 40


def sk_votes(d, X, prefix, thr_file):
    thr = json.load(open(os.path.join(d, thr_file))); V = []
    for n in SK:
        p = os.path.join(d, f"{prefix}{n}.joblib")
        if os.path.exists(p):
            V.append((joblib.load(p).predict_proba(X)[:, 1] >= float(thr.get(n, 0.5))).astype(int))
    return np.vstack(V)


def main():
    df = pd.read_csv(os.path.join(DATA, "yelp_frontend_multimodal.csv")); df["review_text"] = df["review_text"].astype(str)
    dl = pd.read_csv(os.path.join(HERE, "dl_predictions_all.csv"))
    assert len(dl) == len(df) and (dl["ground_truth_label"].values == df["ground_truth_label"].values).all()
    y = df["ground_truth_label"].values.astype(int)
    texts = df["review_text"].tolist()
    mm = pd.DataFrame({"text": texts})
    for c in FEAT:
        mm[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")

    print("voting baseline sklearn ...", flush=True)
    base_sk = sk_votes(os.path.join(DATA, "yelp_fake_sklearn_models"), texts, "yelp_fake_", "yelp_fake_thresholds.json")
    print("voting Model B sklearn ...", flush=True)
    mb_sk = sk_votes(os.path.join(DATA, "yelp_multimodal_full_sklearn_models"), mm, "", "thresholds.json")

    base_M = np.vstack([base_sk, dl["base_lstm"].values, dl["base_db"].values])
    mb_M   = np.vstack([mb_sk,   dl["mb_lstm"].values,   dl["mb_db"].values])
    base_pred = (base_M.sum(0) > base_M.shape[0] / 2).astype(int)
    mb_pred   = (mb_M.sum(0)   > mb_M.shape[0]   / 2).astype(int)

    g = pd.DataFrame({"biz": df["business_id"].values, "y": y,
                      "base_ok": (base_pred == y).astype(int), "mb_ok": (mb_pred == y).astype(int)})
    agg = g.groupby("biz").agg(n=("y", "size"), base=("base_ok", "mean"), mb=("mb_ok", "mean")).reset_index()
    agg = agg[agg.n >= MIN_REVIEWS].copy()
    agg["margin"] = agg["mb"] - agg["base"]
    agg["status"] = np.where(agg.margin > 1e-9, "WIN", np.where(agg.margin < -1e-9, "LOSE", "TIE"))
    agg = agg.sort_values("margin", ascending=False).reset_index(drop=True)

    nW = int((agg.status == "WIN").sum()); nT = int((agg.status == "TIE").sum()); nL = int((agg.status == "LOSE").sum())
    tot = len(agg)
    print("\n" + "=" * 60)
    print(f"Per-business (>= {MIN_REVIEWS} reviews): {tot} businesses")
    print(f"  Model B  WINS {nW} ({nW/tot:.0%})  |  TIES {nT} ({nT/tot:.0%})  |  LOSES {nL} ({nL/tot:.0%})")
    print("=" * 60)

    def block(rows, title):
        print(f"\n{title}  ({len(rows)})")
        print(f"  {'business':<16}{'n':>5}{'baseline':>10}{'ModelB':>9}{'margin':>9}")
        for _, r in rows.iterrows():
            print(f"  {r.biz:<16}{int(r.n):>5}{r.base:>9.0%}{r.mb:>9.0%}{r.margin*100:>+8.0f}")

    block(agg[agg.status == "WIN"],  "MODEL B WINS")
    block(agg[agg.status == "TIE"],  "TIES")
    block(agg[agg.status == "LOSE"].sort_values("margin"), "BASELINE WINS (Model B loses)")

    agg.to_csv(os.path.join(HERE, "per_business_full.csv"), index=False)
    print("\nwrote per_business_full.csv")


if __name__ == "__main__":
    main()
