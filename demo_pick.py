"""
demo_pick.py — pick the best business for a LIVE DEMO that loads only the first N (default 15)
reviews and shows multi-modal (Model B) beating text-only. The frontend serves a business's
reviews sorted by TIME (oldest first), so this scores exactly the first-N served reviews — not the
whole business — and ranks by how many MORE of those N Model B classifies correctly than the
baseline (plus per-review "flips" = Model B right where baseline is wrong). 9-model ensembles
(7 sklearn ex-knn + LSTM + DistilBERT from dl_predictions_all.csv).
RUN: PYTHONIOENCODING=utf-8 python -u demo_pick.py
"""
import os
os.environ.setdefault("USE_TF", "0")
import json, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
import model_b

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
SK = ["rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]
FEAT = model_b.FEAT_COLS
N_DEMO = 15


def sk_votes(d, X, prefix, thr_file):
    thr = json.load(open(os.path.join(d, thr_file))); V = []
    for n in SK:
        p = os.path.join(d, f"{prefix}{n}.joblib")
        if os.path.exists(p):
            V.append((joblib.load(p).predict_proba(X)[:, 1] >= float(thr.get(n, 0.5))).astype(int))
    return np.vstack(V)


def main():
    df = pd.read_csv(os.path.join(DATA, "yelp_frontend_multimodal.csv"))
    df["review_text"] = df["review_text"].astype(str)
    df["time"] = pd.to_numeric(df.get("time", 0), errors="coerce").fillna(0).astype("int64")
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

    w = pd.DataFrame({"biz": df["business_id"].values, "time": df["time"].values, "y": y,
                      "base_ok": (base_pred == y).astype(int), "mb_ok": (mb_pred == y).astype(int)})
    rows = []
    for biz, grp in w.groupby("biz"):
        g = grp.sort_values("time").head(N_DEMO)
        if len(g) < N_DEMO:
            continue
        base_c = int(g.base_ok.sum()); mb_c = int(g.mb_ok.sum())
        flips = int(((g.mb_ok == 1) & (g.base_ok == 0)).sum())     # MB right, baseline wrong
        anti  = int(((g.mb_ok == 0) & (g.base_ok == 1)).sum())     # baseline right, MB wrong
        rows.append({"biz": biz, "fakes_in_15": int(g.y.sum()),
                     "base_correct": base_c, "mb_correct": mb_c,
                     "net": mb_c - base_c, "flips": flips, "anti": anti})
    R = pd.DataFrame(rows)
    # demo value: most extra-correct for Model B, then most flips, then fewest anti-flips,
    # and prefer a mix (>=3 fakes) so the 15 aren't trivially all-genuine.
    R["mix_ok"] = (R.fakes_in_15 >= 3) & (R.fakes_in_15 <= N_DEMO - 3)
    R = R.sort_values(["net", "flips", "mix_ok", "anti"], ascending=[False, False, False, True]).reset_index(drop=True)

    print(f"\nFirst-{N_DEMO} served reviews (time-sorted) — Model B vs baseline correct-count:")
    print(f"  {'business':<34}{'fakes':>6}{'base':>6}{'MB':>5}{'net':>5}{'flips':>7}{'anti':>6}{'mix':>5}")
    for _, r in R.head(15).iterrows():
        print(f"  {r.biz:<34}{r.fakes_in_15:>6}{r.base_correct:>6}{r.mb_correct:>5}{r.net:>+5}"
              f"{r.flips:>7}{r.anti:>6}{'  Y' if r.mix_ok else '  .':>5}")
    R.to_csv(os.path.join(HERE, "demo_pick.csv"), index=False)
    best = R[R.mix_ok].iloc[0] if R.mix_ok.any() else R.iloc[0]
    print(f"\n>>> RECOMMENDED demo business: {best.biz}")
    print(f"    first {N_DEMO}: Model B {best.mb_correct}/{N_DEMO} correct vs baseline {best.base_correct}/{N_DEMO}  "
          f"(+{best.net}); {best.flips} reviews Model B catches that baseline misses; "
          f"{best.fakes_in_15} fakes / {N_DEMO}.")
    print("wrote demo_pick.csv")


if __name__ == "__main__":
    main()
